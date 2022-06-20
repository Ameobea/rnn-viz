import * as tfc from '@tensorflow/tfjs-core';
export * as tfc from '@tensorflow/tfjs-core';
import { type Tensor, type Rank, tidy, type GradSaveFunc } from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';

const applyAmeoInner = (x: Tensor<Rank>) =>
  tidy(() => {
    // if (val <= -1) return Math.max(val + 2, 0);
    // else if (val <= 0) return -val;
    // else return Math.min(val, 1)
    const part1Mask = tfc.lessEqual(x, -1);
    const part1 = tfc.maximum(x.add(2), 0);
    const part2Mask = tfc.lessEqual(x, 0);
    const part2 = x.neg();
    const part3 = tfc.minimum(x, 1);

    return tfc.where(part1Mask, part1, tfc.where(part2Mask, part2, part3));
  });

const ameoGrad = (dy: tfc.Tensor<Rank>, x: tfc.Tensor<Rank>) => {
  // console.log('x:', Array.from(x.dataSync()));
  // console.log('dy:', Array.from(_dy.dataSync()));

  // [-Infinity, -2] and (1, Infinity]
  // const part1Mask = tfc.logicalOr(tfc.lessEqual(x, -2), tfc.greater(x, 1));
  const part1Grad = tfc.zeros(x.shape);
  // (-2, -1] and (0, 1]
  const part2Mask = tfc.logicalOr(
    tfc.lessEqual(x, -1).logicalAnd(tfc.greater(x, -2)),
    tfc.lessEqual(x, 1).logicalAnd(tfc.greater(x, 0))
  );
  const part2Grad = tfc.fill(x.shape, 1);
  // (-1, 0]
  const part3Mask = tfc.lessEqual(x, 0).logicalAnd(tfc.greater(x, -1));
  const part3Grad = tfc.fill(x.shape, -1);

  return tfc.where(part3Mask, part3Grad, tfc.where(part2Mask, part2Grad, part1Grad)).mul(dy);
};

const applyAmeo = tfc.customGrad((x, save) => {
  const tensor = x as Tensor<Rank>;
  (save as GradSaveFunc)([tensor]);
  const y = applyAmeoInner(tensor);
  return { value: y, gradFunc: (dy, saved) => ameoGrad(dy, saved[0]) };
});

export class Ameo extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    return applyAmeo(tensor);
  }
}

const applySoftAmeoInner = (x: Tensor<Rank>) =>
  tidy(() => {
    // if (val <= -2) return 0;
    // else if (val <= -1.5) return 8 * Math.pow(val + 2, 4);
    // // else if (val <= -1)   return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
    // else if (val <= -0.5) return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
    // else if (val <= 0.5) return 8 * Math.pow(val, 4);
    // else if (val <= 1) return -8 * Math.pow(val, 4) + 32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + 32 * val - 7;
    // else return 1;
    throw new Error('TODO');
  });

export class SoftAmeo extends Activation {
  apply(tensor: Tensor<Rank>, axis?: number | undefined): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
}

export class LeakyAmeo extends Activation {
  apply(tensor: Tensor<Rank>, axis?: number | undefined): Tensor<Rank> {
    throw new Error('Method not implemented.');
  }
}
