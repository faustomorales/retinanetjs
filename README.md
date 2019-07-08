# retinanetjs

This package provides some convenience methods for using TensorFlow models created using `keras-retinanet`.

## Getting Started

### Convert RetinaNet Model to TensorFlowJS

In Python, use the two functions defined in `notebooks/Functions.ipynb` to convert an existing RetinaNet model to a RetinaNetJS-compatible TensorFlowJS folder. The only dependencies are `keras_retinanet` and `tensorflowjs`.

The two functions (`save_retinanet_to_savedmodel` and `convert_retinanet_savedmodel_to_tfjs`) have to be used in two separate Python sessions (e.g., in Jupyter, you must restart the kernel).

For example, to convert the pretrained ResNet50 model, you can follow the example in `notebooks/Convert COCO (ResNet50).ipynb`.

### Using Model with retinanetjs
This is the (easier) part. See examples for use cases.

```
import { load } from 'retinanetjs'

detector = await load('http://www.example.com/path/to/coco_resnet50', COCO_CLASSES)
```