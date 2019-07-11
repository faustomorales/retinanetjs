import * as tf from '@tensorflow/tfjs';

/**
 * This duplicates the behavior of `keras_retinanet.utils.anchors.AnchorParameters`
 */
export interface AnchorParameters {
  ratios: number[];
  sizes: number[];
  strides: number[];
  scales: number[];
}

/**
 * This duplicates the behavior of `keras_retinanet.utils.anchors.AnchorParameters.default`
 */
export const defaultAnchorParameters: AnchorParameters = {
  ratios: [0.5, 1, 2],
  scales: [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
  sizes: [32, 64, 128, 256, 512],
  strides: [8, 16, 32, 64, 128]
};

/**
 * This duplicates the behavior of `keras_retinanet.utils.anchors.shift`
 * @hidden
 */
export function shift(
  shape: number[],
  stride: number,
  anchors: tf.Tensor2D
): tf.Tensor2D {
  return tf.tidy(() => {
    const [yLength, xLength] = shape;
    const shiftX = tf
      .range(0, xLength)
      .add(0.5)
      .mul(stride)
      .expandDims(0)
      .tile([yLength, 1])
      .flatten();
    const shiftY = tf
      .range(0, yLength)
      .add(0.5)
      .mul(stride)
      .expandDims(1)
      .tile([1, xLength])
      .flatten();
    const shifts = tf
      .stack([shiftX, shiftY, shiftX, shiftY])
      .transpose() as tf.Tensor2D;

    const A = anchors.shape[0];
    const K = shifts.shape[0];
    return anchors
      .reshape([1, A, 4])
      .add(shifts.reshape([1, K, 4]).transpose([1, 0, 2]))
      .reshape([K * A, 4]);
  });
}

/**
 * This duplicates the behavior of `keras_retinanet.utils.anchors.generate_anchors`
 * @hidden
 */
export function generateAnchors(
  baseSize: number,
  anchorParams = defaultAnchorParameters
): tf.Tensor2D {
  return tf.tidy(() => {
    const numAnchors = anchorParams.ratios.length * anchorParams.scales.length;
    const scales = tf.tensor1d(anchorParams.scales);
    const ratios = tf.tensor1d(anchorParams.ratios);

    // Scale base size
    let w = tf.mul(tf.tile(scales, [anchorParams.ratios.length]), baseSize); // tslint:disable-line
    let h = tf.mul(tf.tile(scales, [anchorParams.ratios.length]), baseSize); // tslint:disable-line

    // Compute areas of anchors
    const area = tf.mul(w, h);

    // Correct for ratios
    w = tf.sqrt(
      tf.div(
        area,
        tf.tile(ratios.expandDims(1), [1, anchorParams.scales.length]).flatten()
      )
    );
    h = tf.mul(
      w,
      tf.tile(ratios.expandDims(1), [1, anchorParams.scales.length]).flatten()
    );

    // transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    const x1 = tf.sub(tf.zeros([numAnchors]), tf.div(w, 2));
    const y1 = tf.sub(tf.zeros([numAnchors]), tf.div(h, 2));
    const x2 = tf.add(x1, w);
    const y2 = tf.add(y1, h);
    return tf.concat([x1, y1, x2, y2].map(t => t.expandDims(1)), 1);
  });
}

/**
 * This duplicates the behavior of `keras_retinanet.utils.anchors.anchors_for_shape`
 * @hidden
 */
export function anchorsForShape(
  inputShape: number[],
  anchorParams = defaultAnchorParameters
) {
  return tf.tidy(() => {
    const pyramidLevels = [3, 4, 5, 6, 7];
    const imageShape = inputShape.slice(0, 2);
    return tf.concat(
      pyramidLevels.map((p, pIdx) => {
        const curImageShape = imageShape.map(s =>
          Math.floor((s + 2 ** p - 1) / 2 ** p)
        );
        const anchors = generateAnchors(anchorParams.sizes[pIdx], anchorParams);
        const shiftedAnchors = shift(
          curImageShape,
          anchorParams.strides[pIdx],
          anchors
        );
        return shiftedAnchors;
      }),
      0
    );
  });
}
