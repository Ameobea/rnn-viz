import { type Tensor, type Rank, type Scalar, tidy, serialization } from '@tensorflow/tfjs';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';

export class QuantizationRegularizer extends Regularizer {
  private quantizationInterval: number;
  private intensity: number;

  constructor(quantizationInterval: number, intensity: number) {
    super();
    this.quantizationInterval = quantizationInterval;
    this.intensity = intensity;
  }

  apply(x: Tensor<Rank>): Scalar {
    return tidy(() => {
      const xQuantized = x.div(this.quantizationInterval);
      const xQuantizedRounded = xQuantized.round();
      const xQuantizationError = xQuantized.sub(xQuantizedRounded);
      const xQuantizationErrorAbs = xQuantizationError.abs();
      const xQuantizationErrorAbsMean = xQuantizationErrorAbs.mean() as Scalar;
      const penalty = xQuantizationErrorAbsMean.mul(this.intensity);
      return penalty as Scalar;
    });
  }

  getConfig(): serialization.ConfigDict {
    return {
      quantizationInterval: this.quantizationInterval,
      intensity: this.intensity,
    };
  }

  static fromConfig(
    cls: serialization.SerializableConstructor<any>,
    config: serialization.ConfigDict
  ): any {
    return new QuantizationRegularizer(config.quantizationInterval as any, config.intensity as any);
  }
}
