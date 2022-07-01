import type { Rank, Tensor } from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';

export class GCUActivation extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    return tensor.cos().mul(tensor);
  }
}
