import * as tfc from '@tensorflow/tfjs-core';
export * as tfc from '@tensorflow/tfjs-core';
import { type Tensor, type Rank, tidy, type GradSaveFunc, engine } from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';
import {
  unaryKernelFunc as unaryGLSLKernelFunc,
  binaryKernelFunc as binaryGLSLKernelFunc,
} from '@tensorflow/tfjs-backend-webgl/dist/kernel_utils/kernel_funcs_utils';
import { unaryKernelFunc as unaryCPUKernelFunc } from '@tensorflow/tfjs-backend-cpu/dist/utils/unary_utils';
import { binaryKernelFunc as binaryCPUKernelFunc } from '@tensorflow/tfjs-backend-cpu/dist/utils/binary_utils';
import { createSimpleBinaryKernelImpl as createSimpleBinaryCPUKernelImpl } from '@tensorflow/tfjs-backend-cpu/dist/utils/binary_impl';

// const applyAmeoInner = (x: Tensor<Rank>) =>
//   tidy(() => {
//     // if (val <= -1) return Math.max(val + 2, 0);
//     // else if (val <= 0) return -val;
//     // else return Math.min(val, 1)
//     const part1Mask = tfc.lessEqual(x, -1);
//     const part1 = tfc.maximum(x.add(2), 0);
//     const part2Mask = tfc.lessEqual(x, 0);
//     const part2 = x.neg();
//     const part3 = tfc.minimum(x, 1);

//     return tfc.where(part1Mask, part1, tfc.where(part2Mask, part2, part3));
//   });

// const ameoGrad = (dy: tfc.Tensor<Rank>, x: tfc.Tensor<Rank>) =>
//   tfc.tidy(() => {
//     // console.log('x:', Array.from(x.dataSync()));
//     // console.log('dy:', Array.from(_dy.dataSync()));

//     // [-Infinity, -2] and (1, Infinity]
//     // const part1Mask = tfc.logicalOr(tfc.lessEqual(x, -2), tfc.greater(x, 1));
//     const part1Grad = tfc.zeros(x.shape);
//     // (-2, -1] and (0, 1]
//     const part2Mask = tfc.logicalOr(
//       tfc.lessEqual(x, -1).logicalAnd(tfc.greater(x, -2)),
//       tfc.lessEqual(x, 1).logicalAnd(tfc.greater(x, 0))
//     );
//     const part2Grad = tfc.fill(x.shape, 1);
//     // (-1, 0]
//     const part3Mask = tfc.lessEqual(x, 0).logicalAnd(tfc.greater(x, -1));
//     const part3Grad = tfc.fill(x.shape, -1);

//     return tfc.where(part3Mask, part3Grad, tfc.where(part2Mask, part2Grad, part1Grad)).mul(dy);
//   });

export class Ameo extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    const y: Tensor<Rank> = engine().runKernel('ameo', { x: tensor.mul(0.5).sub(0.5) });
    return y.sub(0.5).mul(2);
    // return applyAmeo(tensor.mul(0.5).sub(0.5)).sub(0.5).mul(2);
    // return applyAmeo(tensor);
  }
}

const AMEO_GLSL = `
  if (x <= -1.) {
    return max(x + 2., 0.);
  } else if (x <= 0.) {
    return -x;
  } else {
    return min(x, 1.);
  }
`;

tfc.registerKernel({
  kernelName: 'ameo',
  backendName: 'webgl',
  kernelFunc: unaryGLSLKernelFunc({ opSnippet: AMEO_GLSL }),
});

tfc.registerKernel({
  kernelName: 'ameo',
  backendName: 'cpu',
  kernelFunc: unaryCPUKernelFunc('ameo', (x: number) => {
    if (x <= -1) {
      return Math.max(x + 2, 0);
    } else if (x <= 0) {
      return -x;
    } else {
      return Math.min(x, 1);
    }
  }),
});

const AMEO_GRAD_GLSL = `
  if ((x <= -1. && x > -2.) || (x <= 1. && x > 0.)) {
    return 1.;
  } else if (x <= 0. && x > -1.) {
    return -1.;
  } else {
    return 0.;
  }
`;

tfc.registerKernel({
  kernelName: 'ameoGrad',
  backendName: 'webgl',
  kernelFunc: unaryGLSLKernelFunc({ opSnippet: AMEO_GRAD_GLSL }),
});

tfc.registerKernel({
  kernelName: 'ameoGrad',
  backendName: 'cpu',
  kernelFunc: unaryCPUKernelFunc('ameoGrad', (x: number) => {
    if ((x <= -1 && x > -2) || (x <= 1 && x > 0)) {
      return 1;
    } else if (x <= 0 && x > -1) {
      return -1;
    } else {
      return 0;
    }
  }),
});

tfc.registerGradient({
  kernelName: 'ameo',
  inputsToSave: ['x'],
  gradFunc: (dy, saved) => {
    if (Array.isArray(dy)) {
      throw new Error('ameo gradient only supports a single tensor');
    }
    return { x: () => (engine().runKernel('ameoGrad', { x: saved[0] }) as Tensor<Rank>).mul(dy) };
  },
});

const LEAKY_AMEO_GLSL = `
  float x = a;
  float leakyness = b;
  if (x <= -2.) {
    return (x + 2.) * leakyness;
  } if (x <= -1.) {
    return max(x + 2., 0.);
  } else if (x <= 0.) {
    return -x;
  } else if (x <= 1.) {
    return min(x, 1.);
  } else {
    return x * leakyness;
  }
`;

tfc.registerKernel({
  kernelName: 'leakyAmeo',
  backendName: 'webgl',
  kernelFunc: binaryGLSLKernelFunc({ opSnippet: LEAKY_AMEO_GLSL }),
});

tfc.registerKernel({
  kernelName: 'leakyAmeo',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'leakyAmeo',
    createSimpleBinaryCPUKernelImpl((x: number | string, leakyness: number | string) => {
      if (typeof x === 'string' || typeof leakyness === 'string') {
        throw new Error('leakyAmeo only supports number types');
      }

      if (x <= -2) {
        return (x + 2) * leakyness;
      } else if (x <= -1) {
        return Math.max(x + 2, 0);
      } else if (x <= 0) {
        return -x;
      } else if (x <= 1) {
        return Math.min(x, 1);
      } else {
        return 1 + (x - 1) * leakyness;
      }
    })
  ),
});

const LEAKY_AMEO_GRAD_GLSL = `
  if ((x <= -1. && x > -2.) || (x <= 1. && x > 0.)) {
    return 1.;
  } else if (x <= 0. && x > -1.) {
    return -1.;
  else {
    return leakyness;
  }
`;

tfc.registerKernel({
  kernelName: 'leakyAmeoGrad',
  backendName: 'webgl',
  kernelFunc: binaryGLSLKernelFunc({ opSnippet: LEAKY_AMEO_GRAD_GLSL }),
});

tfc.registerKernel({
  kernelName: 'leakyAmeoGrad',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'ameoGrad',
    createSimpleBinaryCPUKernelImpl((x: number | string, leakyness: number | string) => {
      if (typeof x === 'string' || typeof leakyness === 'string') {
        throw new Error('leakyAmeoGrad only supports number types');
      }

      if ((x <= -1 && x > -2) || (x <= 1 && x > 0)) {
        return 1;
      } else if (x <= 0 && x > -1) {
        return -1;
      } else {
        return leakyness;
      }
    })
  ),
});

tfc.registerGradient({
  kernelName: 'leakyAmeo',
  inputsToSave: ['a', 'b'],
  gradFunc: (dy, saved) => {
    if (Array.isArray(dy)) {
      throw new Error('leakyAmeo gradient only supports a single tensor');
    }

    // console.log({ saved: saved.map(t => [...t.dataSync()]), dy: [...dy.dataSync()] });
    const x: Tensor<Rank> = saved[0];
    const leakyness = saved[1];
    return {
      a: () =>
        (engine().runKernel('leakyAmeoGrad', { a: x, b: leakyness }) as Tensor<Rank>)
          .mul(dy)
          .reshape(x.shape),
      b: () => tfc.scalar(0),
    };
  },
});

export class LeakyAmeo extends Activation {
  leakyness: Tensor<Rank>;

  constructor(leakyness: number) {
    super();
    this.leakyness = tfc.tensor1d([leakyness]);
  }

  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    const y: Tensor<Rank> = engine().runKernel('leakyAmeo', {
      a: tensor.mul(0.5).sub(0.5),
      b: this.leakyness,
    });
    return y.sub(0.5).mul(2);
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
    return applySoftAmeo(tensor.mul(0.5).sub(0.5)).sub(0.5).mul(2);
  }
}

// const applySoftLeakyAmeoInner = (x: Tensor<Rank>, leakyness: number) =>
//   tidy(() => {
//     // if (val <= -2) return leakyness * (val + 2);
//     // else if (val <= -1.5) return 8 * Math.pow(val + 2, 4);
//     // // else if (val <= -1)   return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
//     // else if (val <= -0.5) return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
//     // else if (val <= 0.5) return 8 * Math.pow(val, 4);
//     // else if (val <= 1) return -8 * Math.pow(val, 4) + 32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + 32 * val - 7;
//     // else return leakyness * (val - 1) + 1;
//     const part1Mask = tfc.lessEqual(x, -2);
//     const part1 = tfc.fill(x.shape, leakyness).mul(x.add(2));
//     const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
//     const part2 = tfc.fill(x.shape, 8).mul(tfc.pow(x.add(2), 4));
//     const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
//     const part3 = tfc
//       .fill(x.shape, -8)
//       .mul(tfc.pow(x, 4))
//       .add(
//         tfc
//           .fill(x.shape, -32)
//           .mul(tfc.pow(x, 3))
//           .add(
//             tfc
//               .fill(x.shape, -48)
//               .mul(tfc.pow(x, 2))
//               .add(tfc.fill(x.shape, -32).mul(x).add(tfc.fill(x.shape, -7)))
//           )
//       );
//     const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
//     const part4 = tfc.fill(x.shape, 8).mul(tfc.pow(x, 4));
//     const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
//     const part5 = tfc
//       .fill(x.shape, -8)
//       .mul(tfc.pow(x, 4))
//       .add(
//         tfc
//           .fill(x.shape, 32)
//           .mul(tfc.pow(x, 3))
//           .add(
//             tfc
//               .fill(x.shape, -48)
//               .mul(tfc.pow(x, 2))
//               .add(tfc.fill(x.shape, 32).mul(x).add(tfc.fill(x.shape, -7)))
//           )
//       );
//     const part6 = tfc.fill(x.shape, leakyness).mul(x.sub(1)).add(1);

//     return tfc.where(
//       part1Mask,
//       part1,
//       tfc.where(
//         part2Mask,
//         part2,
//         tfc.where(part3Mask, part3, tfc.where(part4Mask, part4, tfc.where(part5Mask, part5, part6)))
//       )
//     );
//   });

// const softLeakyAmeoGrad = (dy: Tensor<Rank>, x: Tensor<Rank>, leakyness: number) =>
//   tfc.tidy(() => {
//     // x <= -2
//     const part1Mask = tfc.lessEqual(x, -2);
//     const part1Grad = tfc.fill(x.shape, leakyness);
//     // x <= -1.5 && x > -2
//     const part2Mask = tfc.logicalAnd(tfc.lessEqual(x, -1.5), tfc.greater(x, -2));
//     const part2Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x.add(2), 3));
//     // x <= -0.5 && x > -1.5
//     const part3Mask = tfc.logicalAnd(tfc.lessEqual(x, -0.5), tfc.greater(x, -1.5));
//     const part3Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.add(1), 3));
//     // x <= 0.5 && x > -0.5
//     const part4Mask = tfc.logicalAnd(tfc.greater(x, -0.5), tfc.lessEqual(x, 0.5));
//     const part4Grad = tfc.fill(x.shape, 32).mul(tfc.pow(x, 3));
//     // x <= 1 && x > 0.5
//     const part5Mask = tfc.logicalAnd(tfc.greater(x, 0.5), tfc.lessEqual(x, 1));
//     const part5Grad = tfc.fill(x.shape, -32).mul(tfc.pow(x.sub(1), 3));
//     // x > 1
//     const part6Grad = tfc.fill(x.shape, leakyness);

//     return tfc
//       .where(
//         part1Mask,
//         part1Grad,
//         tfc.where(
//           part2Mask,
//           part2Grad,
//           tfc.where(
//             part3Mask,
//             part3Grad,
//             tfc.where(part4Mask, part4Grad, tfc.where(part5Mask, part5Grad, part6Grad))
//           )
//         )
//       )
//       .mul(dy);
//   });

// const mkApplySoftLeakyAmeo = (leakyness: number) =>
//   tfc.customGrad((x, save) => {
//     const tensor = x as Tensor<Rank>;
//     (save as GradSaveFunc)([tensor]);
//     const y = applySoftLeakyAmeoInner(tensor, leakyness);
//     return { value: y, gradFunc: (dy, saved) => softLeakyAmeoGrad(dy, saved[0], leakyness) };
//   });

const SOFT_LEAKY_AMEO_GLSL = `
  float x = a;
  float leakyness = b;
  if (x <= -2.) {
    return leakyness * (x + 2.);
  } else if (x <= -1.5) {
    return 8. * pow(x + 2., 4.);
  } else if (x <= -0.5) {
    return -8. * pow(x, 4.) - 32. * pow(x, 3.) - 48. * pow(x, 2.) - 32. * x - 7.;
  } else if (x <= 0.5) {
    return 8. * pow(x, 4.);
  } else if (x <= 1.) {
    return -8. * pow(x, 4.) + 32. * pow(x, 3.) - 48. * pow(x, 2.) + 32. * x - 7.;
  } else {
    return leakyness * (x - 1.) + 1.;
  }
`;

tfc.registerKernel({
  kernelName: 'softLeakyAmeo',
  backendName: 'webgl',
  kernelFunc: binaryGLSLKernelFunc({ opSnippet: SOFT_LEAKY_AMEO_GLSL }),
});

tfc.registerKernel({
  kernelName: 'softLeakyAmeo',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'softLeakyAmeo',
    createSimpleBinaryCPUKernelImpl((x: number | string, leakyness: number | string) => {
      if (typeof x === 'string' || typeof leakyness === 'string') {
        throw new Error('softLeakyAmeo only supports number types');
      }

      if (x <= -2) return leakyness * (x + 2);
      else if (x <= -1.5) return 8 * Math.pow(x + 2, 4);
      else if (x <= -0.5)
        return -8 * Math.pow(x, 4) - 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) - 32 * x - 7;
      else if (x <= 0.5) return 8 * Math.pow(x, 4);
      else if (x <= 1)
        return -8 * Math.pow(x, 4) + 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) + 32 * x - 7;
      else return leakyness * (x - 1) + 1;
    })
  ),
});

const SOFT_LEAKY_AMEO_GRAD_GLSL = `
  float x = a;
  float leakyness = b;
  if (x <= -2.) {
    return leakyness;
  } else if (x <= -1.5) {
    return 32. * pow(x + 2., 3.);
  } else if (x <= -0.5) {
    return -32. * pow(x, 3.) - 96. * pow(x, 2.) - 96. * x - 32.;
  } else if (x <= 0.5) {
    return 32. * pow(x, 3.);
  } else if (x <= 1.) {
    return -32. * pow(x, 3.) + 96. * pow(x, 2.) - 96. * x + 32.;
  } else {
    return leakyness;
  }
`;

tfc.registerKernel({
  kernelName: 'softLeakyAmeoGrad',
  backendName: 'webgl',
  kernelFunc: binaryGLSLKernelFunc({ opSnippet: SOFT_LEAKY_AMEO_GRAD_GLSL }),
});

tfc.registerKernel({
  kernelName: 'softLeakyAmeoGrad',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'softLeakyAmeoGrad',
    createSimpleBinaryCPUKernelImpl((x: number | string, leakyness: number | string) => {
      if (typeof x === 'string' || typeof leakyness === 'string') {
        throw new Error('softLeakyAmeoGrad only supports number types');
      }

      if (x <= -2) return leakyness;
      else if (x <= -1.5) return 32 * Math.pow(x + 2, 3);
      else if (x <= -0.5) return -32 * Math.pow(x, 3) - 96 * Math.pow(x, 2) - 96 * x - 32;
      else if (x <= 0.5) return 32 * Math.pow(x, 3);
      else if (x <= 1) return -32 * Math.pow(x, 3) + 96 * Math.pow(x, 2) - 96 * x + 32;
      else return leakyness;
    })
  ),
});

tfc.registerGradient({
  kernelName: 'softLeakyAmeo',
  inputsToSave: ['a', 'b'],
  gradFunc: (dy, saved) => {
    if (Array.isArray(dy)) {
      throw new Error('softLeakyAmeo gradient only supports a single tensor');
    }

    // console.log({ saved: saved.map(t => [...t.dataSync()]), dy: [...dy.dataSync()] });
    const x: Tensor<Rank> = saved[0];
    const leakyness = saved[1];
    return {
      a: () =>
        (engine().runKernel('softLeakyAmeoGrad', { a: x, b: leakyness }) as Tensor<Rank>)
          .mul(dy)
          .reshape(x.shape),
      b: () => tfc.scalar(0),
    };
  },
});

export class SoftLeakyAmeo extends Activation {
  private leakyness: number;

  constructor(leakyness?: number | null) {
    super();
    this.leakyness = leakyness == null ? 0.01 : leakyness;
  }

  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    // return this.applyInner(tensor.mul(0.5).sub(0.5)).sub(0.5).mul(2);
    const y: Tensor<Rank> = engine().runKernel('softLeakyAmeo', {
      a: tensor.mul(0.5).sub(0.5),
      b: tfc.scalar(this.leakyness),
    });
    return y.sub(0.5).mul(2);
  }
}

// const mkApplyInterpolatedAmeo = (factor: number, leakyness: number) =>

//    tfc.customGrad((x, save) => {
//     const tensor = x as Tensor<Rank>;
//     (save as GradSaveFunc)([tensor]);

//     const y0Mix = factor;
//     const y1Mix = 1 - factor;

//     const y0: Tensor<Rank> = (engine().runKernel('ameo', {
//       x: tensor.mul(0.5).sub(0.5),
//     }) as Tensor<Rank>).mul(y0Mix);
//     const y1: Tensor<Rank> = (engine().runKernel('softLeakyAmeo', {
//       a: tensor.mul(0.5).sub(0.5),
//       b: tfc.scalar(leakyness),
//     }) as Tensor<Rank>).mul(y1Mix);
//     const y = y0.add(y1);

//     return {
//       value: y,
//       gradFunc: (dy, saved) => {
//         const grad0: Tensor<Rank> = (engine().runKernel('ameoGrad', {
//           x: tensor.mul(0.5).sub(0.5),
//         }) as Tensor<Rank>).mul(y0Mix);
//         const grad1: Tensor<Rank> = (engine().runKernel('softLeakyAmeoGrad', {
//           a: tensor.mul(0.5).sub(0.5),
//           b: tfc.scalar(leakyness),
//         }) as Tensor<Rank>).mul(y1Mix);
//         // const grad0: Tensor<Rank> = ameoGrad(dy, saved[0]).mul(y0Mix);
//         // const grad1: Tensor<Rank> = softLeakyAmeoGrad(dy, saved[0], leakyness).mul(y1Mix);
//         return grad0.add(grad1);
//       },
//     };
//   });

/**
 * Interpolation between Ameo and SoftLeakyAmeo
 */
export class InterpolatedAmeo extends Activation {
  // private applyInner: (tensor: Tensor<Rank>) => Tensor<Rank>;
  private factor: number;
  private leakyness: number;

  constructor(factor: number, leakyness: number) {
    if (factor < 0 || factor > 1) {
      throw new Error('`factor` must be between 0 and 1');
    }
    super();
    this.factor = factor;
    this.leakyness = leakyness;
    // this.applyInner = mkApplyInterpolatedAmeo(factor, leakyness);
  }

  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    // return this.applyInner(tensor.mul(0.5).sub(0.5)).sub(0.5).mul(2);
    const y0Mix = this.factor;
    const y1Mix = 1 - this.factor;

    const y0: Tensor<Rank> = (
      engine().runKernel('ameo', {
        x: tensor.mul(0.5).sub(0.5),
      }) as Tensor<Rank>
    ).mul(y0Mix);
    const y1: Tensor<Rank> = (
      engine().runKernel('softLeakyAmeo', {
        a: tensor.mul(0.5).sub(0.5),
        b: tfc.scalar(this.leakyness),
      }) as Tensor<Rank>
    ).mul(y1Mix);
    return y0.add(y1).sub(0.5).mul(2);
  }
}
