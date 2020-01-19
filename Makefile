IMAGE_NAME = retinanetjs
VOLUME_NAME = $(IMAGE_NAME)_env
NOTEBOOK_PORT = 5000
IN_DOCKER = docker run -v $(PWD):/usr/src -v ~/.ssh:/root/.ssh -v $(VOLUME_NAME):/usr/src/node_modules --rm -it $(IMAGE_NAME)

.PHONY: build
build:
	docker build --rm --force-rm -t $(IMAGE_NAME) .
	@-docker volume rm $(VOLUME_NAME)
bash:
	$(IN_DOCKER) bash
init:
	yarn install
download_test_models: build
	mkdir -p test_assets/models/mobilenet224_1_0_oxfordcatdog
	mkdir -p test_assets/models/resnet50_coco_best_v2.1.0
	$(IN_DOCKER) gsutil -m rsync -r -d gs://retinanetjs/models/mobilenet224_1_0_oxfordcatdog ./test_assets/models/mobilenet224_1_0_oxfordcatdog
	$(IN_DOCKER) gsutil -m rsync -r -d gs://retinanetjs/models/resnet50_coco_best_v2.1.0 ./test_assets/models/resnet50_coco_best_v2.1.0
precommit:
	$(IN_DOCKER) yarn test
build-notebooks:
	docker build -t $(IMAGE_NAME)_notebooks -f Dockerfile-notebooks .
.PHONY: notebooks
notebooks: build-notebooks
	docker run -v $(PWD):/usr/src -it -w /usr/src -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME)_notebooks jupyter notebook --ip=0.0.0.0 --port=$(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
prepare-release:
	@-docker volume rm $(VOLUME_NAME)
	$(IN_DOCKER) yarn prepare-release