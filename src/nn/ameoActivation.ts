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

const ameoGrad = (dy: tfc.Tensor<Rank>, x: tfc.Tensor<Rank>) =>
  tfc.tidy(() => {
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
  });

const applyAmeo = tfc.customGrad((x, save) => {
  const tensor = x as Tensor<Rank>;
  (save as GradSaveFunc)([tensor]);
  const y = applyAmeoInner(tensor);
  return { value: y, gradFunc: (dy, saved) => ameoGrad(dy, saved[0]) };
});

export class Ameo extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    return applyAmeo(tensor.mul(1.5).sub(0.5)).sub(0.5).mul(2);
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
    const part1Mask = tfc.lessEqual(x, -2);
    const part1 = tfc.zeros(x.shape);
    const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
    const part2 = tfc.fill(x.shape, 8).mul(tfc.pow(x.add(2), 4));
    const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
    const part3 = tfc
      .fill(x.shape, -8)
      .mul(tfc.pow(x, 4))
      .add(
        tfc
          .fill(x.shape, -32)
          .mul(tfc.pow(x, 3))
          .add(
            tfc
              .fill(x.shape, -48)
              .mul(tfc.pow(x, 2))
              .add(tfc.fill(x.shape, -32).mul(x).add(tfc.fill(x.shape, -7)))
          )
      );
    const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
    const part4 = tfc.fill(x.shape, 8).mul(tfc.pow(x, 4));
    const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
    const part5 = tfc
      .fill(x.shape, -8)
      .mul(tfc.pow(x, 4))
      .add(
        tfc
          .fill(x.shape, 32)
          .mul(tfc.pow(x, 3))
          .add(
            tfc
              .fill(x.shape, -48)
              .mul(tfc.pow(x, 2))
              .add(tfc.fill(x.shape, 32).mul(x).add(tfc.fill(x.shape, -7)))
          )
      );
    const part6 = tfc.fill(x.shape, 1);

    return tfc.where(
      part1Mask,
      part1,
      tfc.where(
        part2Mask,
        part2,
        tfc.where(part3Mask, part3, tfc.where(part4Mask, part4, tfc.where(part5Mask, part5, part6)))
      )
    );
  });

const softAmeoGrad = (dy: Tensor<Rank>, x: Tensor<Rank>) =>
  tfc.tidy(() => {
    // x <= -2
    const part1Mask = tfc.lessEqual(x, -2);
    const part1Grad = tfc.zeros(x.shape);
    // x <= -1.5 && x > -2
    const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
    const part2Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x.add(2), 3));
    // x <= -0.5 && x > -1.5
    const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
    const part3Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.add(1), 3));
    // x <= 0.5 && x > -0.5
    const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
    const part4Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x, 3));
    // x <= 1 && x > 0.5
    const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
    const part5Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.sub(1), 3));
    const part6Grad = tfc.fill(x.shape, 0);

    return tfc
      .where(
        part1Mask,
        part1Grad,
        tfc.where(
          part2Mask,
          part2Grad,
          tfc.where(
            part3Mask,
            part3Grad,
            tfc.where(part4Mask, part4Grad, tfc.where(part5Mask, part5Grad, part6Grad))
          )
        )
      )
      .mul(dy);
  });

const applySoftAmeo = tfc.customGrad((x, save) => {
  const tensor = x as Tensor<Rank>;
  (save as GradSaveFunc)([tensor]);
  const y = applySoftAmeoInner(tensor);
  return { value: y, gradFunc: (dy, saved) => softAmeoGrad(dy, saved[0]) };
});

export class SoftAmeo extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    return applySoftAmeo(tensor.mul(1.5).sub(0.5)).sub(0.5).mul(2);
  }
}

const applySoftLeakyAmeoInner = (x: Tensor<Rank>, leakyness: number) =>
  tidy(() => {
    // if (val <= -2) return leakyness * (val + 2);
    // else if (val <= -1.5) return 8 * Math.pow(val + 2, 4);
    // // else if (val <= -1)   return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
    // else if (val <= -0.5) return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
    // else if (val <= 0.5) return 8 * Math.pow(val, 4);
    // else if (val <= 1) return -8 * Math.pow(val, 4) + 32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + 32 * val - 7;
    // else return leakyness * (val - 1) + 1;
    const part1Mask = tfc.lessEqual(x, -2);
    const part1 = tfc.fill(x.shape, leakyness).mul(x.add(2));
    const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
    const part2 = tfc.fill(x.shape, 8).mul(tfc.pow(x.add(2), 4));
    const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
    const part3 = tfc
      .fill(x.shape, -8)
      .mul(tfc.pow(x, 4))
      .add(
        tfc
          .fill(x.shape, -32)
          .mul(tfc.pow(x, 3))
          .add(
            tfc
              .fill(x.shape, -48)
              .mul(tfc.pow(x, 2))
              .add(tfc.fill(x.shape, -32).mul(x).add(tfc.fill(x.shape, -7)))
          )
      );
    const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
    const part4 = tfc.fill(x.shape, 8).mul(tfc.pow(x, 4));
    const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
    const part5 = tfc
      .fill(x.shape, -8)
      .mul(tfc.pow(x, 4))
      .add(
        tfc
          .fill(x.shape, 32)
          .mul(tfc.pow(x, 3))
          .add(
            tfc
              .fill(x.shape, -48)
              .mul(tfc.pow(x, 2))
              .add(tfc.fill(x.shape, 32).mul(x).add(tfc.fill(x.shape, -7)))
          )
      );
    const part6 = tfc.fill(x.shape, leakyness).mul(x.sub(1)).add(1);

    return tfc.where(
      part1Mask,
      part1,
      tfc.where(
        part2Mask,
        part2,
        tfc.where(part3Mask, part3, tfc.where(part4Mask, part4, tfc.where(part5Mask, part5, part6)))
      )
    );
  });

const softLeakyAmeoGrad = (dy: Tensor<Rank>, x: Tensor<Rank>, leakyness: number) =>
  tfc.tidy(() => {
    // x <= -2
    const part1Mask = tfc.lessEqual(x, -2);
    const part1Grad = tfc.fill(x.shape, leakyness);
    // x <= -1.5 && x > -2
    const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
    const part2Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x.add(2), 3));
    // x <= -0.5 && x > -1.5
    const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
    const part3Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.add(1), 3));
    // x <= 0.5 && x > -0.5
    const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
    const part4Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x, 3));
    // x <= 1 && x > 0.5
    const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
    const part5Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.sub(1), 3));
    // x > 1
    const part6Grad = tfc.fill(x.shape, leakyness);

    return tfc
      .where(
        part1Mask,
        part1Grad,
        tfc.where(
          part2Mask,
          part2Grad,
          tfc.where(
            part3Mask,
            part3Grad,
            tfc.where(part4Mask, part4Grad, tfc.where(part5Mask, part5Grad, part6Grad))
          )
        )
      )
      .mul(dy);
  });

const mkApplySoftLeakyAmeo = (leakyness: number) =>
  tfc.customGrad((x, save) => {
    const tensor = x as Tensor<Rank>;
    (save as GradSaveFunc)([tensor]);
    const y = applySoftLeakyAmeoInner(tensor, leakyness);
    return { value: y, gradFunc: (dy, saved) => softLeakyAmeoGrad(dy, saved[0], leakyness) };
  });

export class SoftLeakyAmeo extends Activation {
  private leakyness: number;
  private applyInner: (tensor: Tensor<Rank>) => Tensor<Rank>;

  constructor(leakyness?: number | null) {
    super();
    this.leakyness = leakyness == null ? 0.01 : leakyness;
    this.applyInner = mkApplySoftLeakyAmeo(this.leakyness);
  }

  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    return this.applyInner(tensor.mul(1.5).sub(0.5)).sub(0.5).mul(2);
  }
}
