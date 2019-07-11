import * as tf from '@tensorflow/tfjs';
import { Zeros } from '@tensorflow/tfjs-layers/dist/initializers'; // tslint:disable-line

import {
  AnchorParameters,
  anchorsForShape,
  defaultAnchorParameters
} from './anchors';

class UpsampleLike extends tf.layers.Layer {
  constructor() {
    super({});
  }

  public computeOutputShape(inputShape: number[][]) {
    return [
      inputShape[0][0],
      inputShape[1][1],
      inputShape[1][2],
      inputShape[0][3]
    ];
  }

  public call(inputs: tf.Tensor4D[], _: object) {
    const [source, target] = inputs;
    const targetShape = target.shape;
    return tf.image.resizeNearestNeighbor(source, [
      targetShape[1],
      targetShape[2]
    ]);
  }

  public static get className() {
    return 'UpsampleLike';
  }
}

class PriorProbability extends Zeros {
  // tslint:disable-line
  /** @nocollapse */
  public static className = 'PriorProbability';
}

tf.serialization.registerClass(UpsampleLike); // Needed for serialization.
tf.serialization.registerClass(PriorProbability);

/**
 * Represents a detected object with coordinates being provided
 * as percentages of the image width and height.
 */
export interface DetectedObject {
  label: string;
  score: number;
  x1: number;
  x2: number;
  y1: number;
  y2: number;
}

/**
 * Represents a RetinaNet model. Rather than creating directly,
 * it is intended to be created using `load()`.
 */
export class RetinaNet {
  public readonly model: tf.LayersModel;
  protected readonly classes: string[];
  protected readonly preprocessingMode: string;
  protected readonly anchorParams: AnchorParameters;
  protected readonly height: number;
  protected readonly width: number;

  constructor(
    model: tf.LayersModel,
    classes: string[],
    preprocessingMode: string,
    anchorParams = defaultAnchorParameters
  ) {
    const [height, width] = model.inputs[0].shape!.slice(1, 3) as number[];
    // tslint:disable-next-line
    if (height === -1 || width === -1) {
      throw new Error('RetinaNetJS only supports fixed input sizes.');
    }
    // tslint:disable-next-line
    if (preprocessingMode !== 'tf' && preprocessingMode !== 'caffe') {
      throw new Error('preprocessingMode must be either `tf` or `caffe`.');
    }
    this.width = width;
    this.height = height;
    this.model = model;
    this.classes = classes;
    this.preprocessingMode = preprocessingMode;
    this.anchorParams = anchorParams;
  }

  /**
   * Computes predictions. We currently do not support class-specific filtering.
   * When non-max suppression is applied, it will be across all boxes, regardless of class.
   *
   * @param img The image object on which to run object detection
   * @param threshold The prediction threshold
   * @param nmsThreshold The non-max suppresion IoU threshold
   */
  public async detect(
    img:
      | tf.Tensor3D
      | ImageData
      | HTMLImageElement
      | HTMLCanvasElement
      | HTMLVideoElement,
    threshold = 0.5,
    nmsThreshold = 0.5
  ): Promise<DetectedObject[]> {
    // Build model input from image
    const [X, padX, padY] = tf.tidy(() => {
      const imageTensor = !(img instanceof tf.Tensor)
        ? tf.browser.fromPixels(img)
        : img;
      return this.handleImageTensor(imageTensor);
    });
    // Run inference
    const y = this.model.predict(X) as tf.Tensor4D[];
    const [coords, classification] = tf.tidy(() => {
      const [boxDeltas, classScores] = y.map(t => t.squeeze([0]));
      const anchorBoxes = anchorsForShape(
        this.model.inputs[0].shape!.slice(1, 3) as number[],
        this.anchorParams
      );

      const mean = [0, 0, 0, 0];
      const std = [0.2, 0.2, 0.2, 0.2];
      const width = anchorBoxes
        .slice([0, 2], [-1, 1])
        .sub(anchorBoxes.slice([0, 0], [-1, 1]));
      const height = anchorBoxes
        .slice([0, 2], [-1, 1])
        .sub(anchorBoxes.slice([0, 0], [-1, 1]));

      const x1y1x2y2 = tf.concat(
        [0, 1, 2, 3].map(i => {
          return anchorBoxes.slice([0, i], [-1, 1]).add(
            boxDeltas
              .slice([0, i], [-1, 1])
              .mul(std[i])
              .add(mean[i])
              .mul(i % 2 === 0 ? width : height)
          );
        }),
        1
      ) as tf.Tensor2D;

      // TODO: Add support for class-specific filtering and turning nms on and off, just
      // like in retinanet.
      return [x1y1x2y2, classScores];
    });
    const selected = await tf.image.nonMaxSuppressionAsync(
      coords,
      classification.max(1, false),
      300,
      nmsThreshold,
      threshold
    );
    const detections = tf.tidy(() => {
      const classificationNms = classification.gather(selected) as tf.Tensor2D;
      const coordsNms = coords
        .gather(selected)
        .div([
          this.width - padX,
          this.height - padY,
          this.width - padX,
          this.height - padY
        ]);
      const x1y1x2y2ls = tf
        .concat(
          [
            coordsNms,
            classificationNms
              .argMax(1)
              .expandDims(1)
              .cast('float32'),
            classificationNms.max(1, true)
          ],
          1
        )
        .arraySync() as number[][];
      return x1y1x2y2ls.map(r => {
        const [x1, y1, x2, y2, labelIdx, score] = r;
        return { label: this.classes[labelIdx], score, x1, x2, y1, y2 };
      });
    });
    X.dispose();
    y.map(t => t.dispose());
    selected.dispose();
    return detections;
  }

  /**
   * Remove the model from memory.
   */
  public dispose() {
    this.model.dispose();
  }

  private handleImageTensor(
    imageTensor: tf.Tensor3D
  ): [tf.Tensor4D, number, number] {
    return tf.tidy(() => {
      const inputHeight = imageTensor.shape[0];
      const inputWidth = imageTensor.shape[1];
      const [outputHeight, outputWidth] = [this.height, this.width];
      const scale = Math.min(
        outputHeight / inputHeight,
        outputWidth / inputWidth
      );
      const padY = outputHeight - Math.round(scale * inputHeight);
      const padX = outputWidth - Math.round(scale * inputWidth);
      const scaledTensor =
        scale === 1
          ? imageTensor
          : imageTensor.resizeBilinear([
              Math.round(scale * inputHeight),
              Math.round(scale * inputWidth)
            ]);
      const paddedTensor =
        padX === 0 && padY === 0
          ? scaledTensor
          : scaledTensor.pad([[0, padY], [0, padX], [0, 0]]);
      const normedTensor =
        this.preprocessingMode === 'tf'
          ? paddedTensor.sub(127.5).div(127.5)
          : paddedTensor.sub(tf.tensor3d([[[103.939, 116.779, 123.68]]]));
      return [normedTensor.expandDims(0).cast('float32'), padX, padY] as [
        tf.Tensor4D,
        number,
        number
      ];
    });
  }
}

/**
 *
 * @param modelPath The path to the model or a `tf.io.IOHandler` object
 * @param classes The list of detected classes
 * @param preprocessingMode One of `tf` or `caffe`. Check the `preprocess_images`
 *   method of your backbone to see which you should use.
 * @param onProgress A callback to report progress
 * @param anchorParams The anchor parameters for your model
 */
export async function load(
  modelPath: string | tf.io.IOHandler,
  classes: string[],
  preprocessingMode: string,
  onProgress?: (progress: number, message: string) => void,
  anchorParams = defaultAnchorParameters
) {
  const tfOnProgress = (progress: number) =>
    onProgress(0.9 * progress, 'Downloading');
  const model = await tf.loadLayersModel(modelPath, {
    onProgress: tfOnProgress,
    strict: false
  });
  const detector = new RetinaNet(
    model,
    classes,
    preprocessingMode,
    anchorParams
  );
  if (onProgress) onProgress(0.92, 'Building'); // tslint:disable-line
  setTimeout(async () => {
    await detector.detect(tf.ones([100, 100, 3]));
    if (onProgress) onProgress(1.0, 'Finished'); // tslint:disable-line
  }, 100);
  return detector;
}
