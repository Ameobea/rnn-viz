import { tidy, type Rank, type Scalar, type Tensor, addN, serialization } from '@tensorflow/tfjs';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';

export class ComposedRegularizer extends Regularizer {
  private regularizers: Regularizer[];

  constructor(...regularizers: Regularizer[]) {
    super();
    this.regularizers = regularizers;
  }

  apply(x: Tensor<Rank>): Scalar {
    return tidy(() => {
      const penalties = this.regularizers.map(r => r.apply(x));
      return addN(penalties);
    });
  }

  getConfig(): serialization.ConfigDict {
    return { regularizers: this.regularizers.map(r => r.getConfig()) };
  }

  static fromConfig(
    cls: serialization.SerializableConstructor<any>,
    config: serialization.ConfigDict
  ): any {
    if (!Array.isArray(config.regularizers)) {
      throw new Error(
        `Invalid config for ComposedRegularizer: ${JSON.stringify(config)}. ` +
          "The 'regularizers' field must be an Array."
      );
    }

    throw new Error('Unimplemented');
  }
}
