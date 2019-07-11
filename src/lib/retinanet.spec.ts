import test from 'ava';
import { createCanvas, loadImage } from 'canvas';

import * as tf from '@tensorflow/tfjs';
import * as tfn from '@tensorflow/tfjs-node';

import { load } from './retinanet';

const COCO_CLASSES = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush'
];

async function imageFilepathToTensor(filepath: string): Promise<tf.Tensor3D> {
  const image = await loadImage(filepath);
  const canvas = await createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  return tf.cast(tf.browser.fromPixels((await canvas) as any), 'float32');
}

test('cats and dogs model', async t => {
  const detector = await load(
    tfn.io.fileSystem(
      'test_assets/models/mobilenet224_1_0_oxfordcatdog/model.json'
    ),
    ['DOG', 'CAT'],
    'tf'
  );
  const catDetections = await detector.detect(
    await imageFilepathToTensor('test_assets/images/cat.jpg'),
    0.4
  );
  const dogDetections = await detector.detect(
    await imageFilepathToTensor('test_assets/images/dog.jpg'),
    0.4
  );
  t.is(catDetections[0].label, 'CAT');
  t.is(dogDetections[0].label, 'DOG');
  t.pass();
});

test('resnet50 model', async t => {
  const detector = await load(
    tfn.io.fileSystem(
      'test_assets/models/resnet50_coco_best_v2.1.0/model.json'
    ),
    COCO_CLASSES,
    'caffe'
  );
  const dogDetections = await detector.detect(
    await imageFilepathToTensor('test_assets/images/dog.jpg'),
    0.4
  );
  t.is(dogDetections[0].label, 'dog');
  t.pass();
});
