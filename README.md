# retinanetjs

This package provides some convenience methods for using TensorFlow models created using `keras-retinanet`. Check out [the docs](https://faustomorales.github.io/retinanetjs/). You can also check out the [example app](https://github.com/faustomorales/retinanetjs-example-app).

## Getting Started

### Convert RetinaNet Model to TensorFlowJS
As an example, we'll convert the ResNet50 weights to TensorFlow.js format. You must have `tensorflowjs` installed.

First, save a fixed input size training model to a Keras h5 file with both the weights and architecture. You **must** supply a fixed input shape. In experimenting with different backbones, only a few functioned correctly with undefined input shapes when loaded with TensorFlowJS.

Importantly, we **do not** convert to a prediction model. Rather, we do the necessary box regression in TensorFlowJS. Including them made some backbones load incorrectly in TensorFlow.js.

```python
import urllib.request

import keras
from keras_retinanet import models

urllib.request.urlretrieve(
    "https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5",
    "resnet50_coco_best_v2.1.0.h5"
)

model = models.backbone(
    backbone_name='resnet50').retinanet(
    num_classes=80,
    inputs=keras.layers.Input((512, 512, 3)
)
)
model.load_weights('resnet50_coco_best_v2.1.0.h5')
model.save('resnet50_coco_best_v2.1.0_full.h5')
```

Then, at the command line, execute the following.

```shell
tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model \
     resnet50_coco_best_v2.1.0_full.h5 \
     resnet50_coco_best_v2.1.0
```

### Using Model with retinanetjs
The code below loads the above model and does detection. We assume that you have a reference to an `HTMLImage` object in `imageRef` and a list of the COCO class label names in `COCO_CLASSES`. Note that you must supply the preprocessing mode. Check the `preprocess_image` method on your backbone to see whether your model uses `tf` or `caffe` preprocessing.

```javascript
import { load } from 'retinanetjs'

const detector = await load(
    'http://www.example.com/path/to/resnet50_coco_best_v2.1.0',
    COCO_CLASSES, "caffe"
)
const detections = detector.detect(imageRef)
```

## Limitations
The following features are not supported at this time:
- Unspecified input shapes (e.g., `(None, None, 3)`)
- Class-specific filtering. For the moment, non-max suppression is performed across all classes.