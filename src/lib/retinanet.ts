import * as tf from '@tensorflow/tfjs';

export interface DetectedObject {
  label: string;
  score: number;
  x1: number;
  x2: number;
  y1: number;
  y2: number;
}

export class RetinaNet {
  protected readonly normalizationMode: string;
  protected readonly classes: string[];
  protected readonly maxSize: number;
  protected readonly model: tf.GraphModel;

  constructor(model: tf.GraphModel, classes: string[], maxSize: number) {
    this.model = model;
    this.classes = classes;
    this.maxSize = maxSize;
  }

  public async detect(
    img:
      | tf.Tensor3D
      | ImageData
      | HTMLImageElement
      | HTMLCanvasElement
      | HTMLVideoElement,
    threshold = 0.5
  ): Promise<DetectedObject[]> {
    // TODO: Allow for fixed size input models.

    // Build model input from image
    const [X, scale] = tf.tidy(() => {
      const imageTensor = !(img instanceof tf.Tensor)
        ? tf.browser.fromPixels(img)
        : img;
      // Reshape to a single-element batch so we can pass it to executeAsync.
      const height = imageTensor.shape[0];
      const width = imageTensor.shape[1];
      const scaleValue: number =
        this.maxSize !== -1 && (height > this.maxSize || width > this.maxSize)
          ? this.maxSize / Math.max(height, width)
          : 1;
      return [
        scaleValue < 1
          ? imageTensor
              .resizeBilinear([
                Math.round(scaleValue * height),
                Math.round(scaleValue * width)
              ])
              .expandDims(0)
          : imageTensor.expandDims(0),
        scaleValue
      ];
    });

    // Run inference
    const y = (await this.model.executeAsync({ images: X }, [
      'retinanet-bbox/lambda_1/output_boxes',
      'retinanet-bbox/lambda_2/output_scores',
      'retinanet-bbox/lambda_3/output_labels'
    ])) as tf.Tensor3D[];

    // Compute detections
    const results = tf.tidy(() => {
      const [boxes, scores, labels] = y;
      return tf
        .concat(
          [
            boxes.mul(scale),
            scores.expandDims(2),
            labels.expandDims(2).cast('float32')
          ],
          2
        )
        .squeeze([0])
        .arraySync() as number[][];
    });
    X.dispose();
    y.map(async t => t.dispose());

    // Build detections list
    const detections: DetectedObject[] = [];
    for (const result of results) {
      const [x1, y1, x2, y2, score, labelIndex] = result;
      // tslint:disable-next-line
      if (score < threshold) {
        // The detections are presorted by the network.
        // We do not need to continue.
        break;
      } else {
        detections.push({
          label: this.classes[labelIndex],
          score,
          x1,
          x2,
          y1,
          y2
        });
      }
    }
    return detections;
  }

  public dispose() {
    this.model ? this.model.dispose() : null; // tslint:disable-line
  }
}

export async function load(
  modelPath: string | tf.io.IOHandler,
  classes: string[],
  maxSize = -1
) {
  const model = await tf.loadGraphModel(modelPath);
  return new RetinaNet(model, classes, maxSize);
}
