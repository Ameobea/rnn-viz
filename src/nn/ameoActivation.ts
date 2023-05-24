import * as tfc from '@tensorflow/tfjs-core';
export * as tfc from '@tensorflow/tfjs-core';
import {
  type Tensor,
  type Rank,
  tidy,
  type GradSaveFunc,
  engine,
  type TypedArray,
  type DataType,
  type NamedTensorInfoMap,
  type NamedAttrMap,
} from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';
import {
  unaryKernelFunc as unaryGLSLKernelFunc,
  binaryKernelFunc as binaryGLSLKernelFunc,
} from '@tensorflow/tfjs-backend-webgl/dist/kernel_utils/kernel_funcs_utils';
import { unaryKernelFunc as unaryCPUKernelFunc } from '@tensorflow/tfjs-backend-cpu/dist/utils/unary_utils';
import { binaryKernelFunc as binaryCPUKernelFunc } from '@tensorflow/tfjs-backend-cpu/dist/utils/binary_utils';
import { createSimpleBinaryKernelImpl as createSimpleBinaryCPUKernelImpl } from '@tensorflow/tfjs-backend-cpu/dist/utils/binary_impl';

export class Ameo extends Activation {
  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    const y: Tensor<Rank> = engine().runKernel('ameo', { x: tensor /*.mul(0.5).sub(0.5)*/ });
    return y;
  }
}

const AMEO_GLSL = `
  x *= 0.5;
  x -= 0.5;

  float y = 0.;
  if (x <= -1.) {
    y = max(x + 2., 0.);
  } else if (x <= 0.) {
    y = -x;
  } else {
    y = min(x, 1.);
  }
  return (y - 0.5) * 2.;
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
    x *= 0.5;
    x -= 0.5;

    let y: number;
    if (x <= -1) {
      y = Math.max(x + 2, 0);
    } else if (x <= 0) {
      y = -x;
    } else {
      y = Math.min(x, 1);
    }
    return (y - 0.5) * 2;
  }),
});

const AMEO_GRAD_GLSL = `
  x *= 0.5;
  x -= 0.5;

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
    x *= 0.5;
    x -= 0.5;

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
    'leakyAmeoGrad',
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

  constructor(leakyness = 0.05) {
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

const SOFT_LEAKY_AMEO_GLSL = `
  float x = a;
  x *= 0.5;
  x -= 0.5;

  float leakyness = b;
  float y = 0.;
  if (x <= -2.) {
    y = leakyness * (x + 2.);
  } else if (x <= -1.5) {
    y = 8. * pow(x + 2., 4.);
  } else if (x <= -0.5) {
    y = -8. * pow(x, 4.) - 32. * pow(x, 3.) - 48. * pow(x, 2.) - 32. * x - 7.;
  } else if (x <= 0.5) {
    y = 8. * pow(x, 4.);
  } else if (x <= 1.) {
    y = -8. * pow(x, 4.) + 32. * pow(x, 3.) - 48. * pow(x, 2.) + 32. * x - 7.;
  } else {
    y = leakyness * (x - 1.) + 1.;
  }
  return (y - 0.5) * 2.;
`;

tfc.registerKernel({
  kernelName: 'softLeakyAmeo',
  backendName: 'webgl',
  kernelFunc: binaryGLSLKernelFunc({ opSnippet: SOFT_LEAKY_AMEO_GLSL }),
});

const nativeSoftLeakyAmeoImpl = createSimpleBinaryCPUKernelImpl(
  (x: number | string, leakyness: number | string) => {
    if (typeof x === 'string' || typeof leakyness === 'string') {
      throw new Error('softLeakyAmeo only supports number types');
    }

    x *= 0.5;
    x -= 0.5;

    let y: number;
    if (x <= -2) y = leakyness * (x + 2);
    else if (x <= -1.5) {
      const xPlus2 = x + 2;
      // y = 8 * Math.pow(x + 2, 4);
      y = 8 * (xPlus2 * xPlus2 * xPlus2 * xPlus2);
    } else if (x <= -0.5) {
      // y = -8 * Math.pow(x, 4) - 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) - 32 * x - 7;
      y = -8 * (x * x * x * x) - 32 * (x * x * x) - 48 * (x * x) - 32 * x - 7;
    } else if (x <= 0.5) {
      // y = 8 * Math.pow(x, 4);
      y = 8 * (x * x * x * x);
    } else if (x <= 1) {
      // y = -8 * Math.pow(x, 4) + 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) + 32 * x - 7;
      y = -8 * (x * x * x * x) + 32 * (x * x * x) - 48 * (x * x) + 32 * x - 7;
    } else y = leakyness * (x - 1) + 1;

    return (y - 0.5) * 2;
  }
);

let wasmEngine: typeof import('../engineComp/engine') | null = null;

export const setWasmEngine = (engine: typeof import('../engineComp/engine')) => {
  wasmEngine = engine;
};

const softLeakyAmeoWasmImpl = (
  wasmEngine: typeof import('../engineComp/engine'),
  leakyness: number,
  xs: Float32Array
): Float32Array => wasmEngine.apply_batch_soft_leaky_ameo(leakyness, xs);

tfc.registerKernel({
  kernelName: 'softLeakyAmeo',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'softLeakyAmeo',
    (
      aShape: number[],
      bShape: number[],
      aVals: TypedArray | string[],
      bVals: TypedArray | string[],
      dtype: DataType
    ): [TypedArray, number[]] => {
      if (wasmEngine) {
        // We assume leakyness is constant for all neurons
        const ys = softLeakyAmeoWasmImpl(wasmEngine, bVals[0] as number, aVals as Float32Array);
        return [ys, aShape];
      }

      return nativeSoftLeakyAmeoImpl(aShape, bShape, aVals, bVals, dtype);
    }
  ),
});

const SOFT_LEAKY_AMEO_GRAD_GLSL = `
  float x = a;
  x *= 0.5;
  x -= 0.5;

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

const softLeakyAmeoGradNativeImpl = createSimpleBinaryCPUKernelImpl(
  (x: number | string, leakyness: number | string) => {
    if (typeof x === 'string' || typeof leakyness === 'string') {
      throw new Error('softLeakyAmeoGrad only supports number types');
    }

    x *= 0.5;
    x -= 0.5;

    if (x <= -2) return leakyness;
    else if (x <= -1.5) {
      // return 32 * Math.pow(x + 2, 3);
      const xPlus2 = x + 2;
      return 32 * (xPlus2 * xPlus2 * xPlus2);
    } else if (x <= -0.5) {
      // return -32 * Math.pow(x, 3) - 96 * Math.pow(x, 2) - 96 * x - 32;
      return -32 * (x * x * x) - 96 * (x * x) - 96 * x - 32;
    } else if (x <= 0.5) {
      // return 32 * Math.pow(x, 3);
      return 32 * (x * x * x);
    } else if (x <= 1) {
      // return -32 * Math.pow(x, 3) + 96 * Math.pow(x, 2) - 96 * x + 32;
      return -32 * (x * x * x) + 96 * (x * x) - 96 * x + 32;
    } else return leakyness;
  }
);

const softLeakyAmeoGradWasmImpl = (
  wasmEngine: typeof import('../engineComp/engine'),
  leakyness: number,
  xs: Float32Array
): Float32Array => wasmEngine.apply_batch_soft_leaky_ameo_grad(leakyness, xs);

tfc.registerKernel({
  kernelName: 'softLeakyAmeoGrad',
  backendName: 'cpu',
  kernelFunc: binaryCPUKernelFunc(
    'softLeakyAmeoGrad',
    (
      aShape: number[],
      bShape: number[],
      aVals: TypedArray | string[],
      bVals: TypedArray | string[],
      dtype: DataType
    ): [TypedArray, number[]] => {
      if (wasmEngine) {
        // We assume leakyness is constant for all neurons
        const ys = softLeakyAmeoGradWasmImpl(wasmEngine, bVals[0] as number, aVals as Float32Array);
        return [ys, aShape];
      }

      return softLeakyAmeoGradNativeImpl(aShape, bShape, aVals, bVals, dtype);
    }
  ),
});

tfc.registerGradient({
  kernelName: 'softLeakyAmeo',
  inputsToSave: ['a', 'b'],
  gradFunc: (dy, saved) => {
    if (Array.isArray(dy)) {
      throw new Error('softLeakyAmeo gradient only supports a single tensor');
    }

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
    const y: Tensor<Rank> = engine().runKernel('softLeakyAmeo', {
      a: tensor,
      b: tfc.scalar(this.leakyness),
    });
    return y;
  }
}

const nativeFusedInterpolatedAmeoImplInner = (
  factor: number,
  x: number | string,
  leakyness: number | string
): number => {
  if (typeof x === 'string' || typeof leakyness === 'string') {
    throw new Error('nativeFusedInterpolatedAmeoImpl only supports number types');
  }

  x *= 0.5;
  x -= 0.5;

  let ameoY: number;
  if (x <= -1) {
    ameoY = Math.max(x + 2, 0);
  } else if (x <= 0) {
    ameoY = -x;
  } else {
    ameoY = Math.min(x, 1);
  }
  ameoY = (ameoY - 0.5) * 2;

  let softLeakyAmeoY: number;
  if (x <= -2) softLeakyAmeoY = leakyness * (x + 2);
  else if (x <= -1.5) {
    // softLeakyAmeoY = 8 * Math.pow(x + 2, 4);
    const xPlus2 = x + 2;
    softLeakyAmeoY = 8 * (xPlus2 * xPlus2 * xPlus2 * xPlus2);
  } else if (x <= -0.5) {
    // softLeakyAmeoY = -8 * Math.pow(x, 4) - 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) - 32 * x - 7;
    softLeakyAmeoY = -8 * (x * x * x * x) - 32 * (x * x * x) - 48 * (x * x) - 32 * x - 7;
  } else if (x <= 0.5) {
    // softLeakyAmeoY = 8 * Math.pow(x, 4);
    softLeakyAmeoY = 8 * (x * x * x * x);
  } else if (x <= 1) {
    // softLeakyAmeoY = -8 * Math.pow(x, 4) + 32 * Math.pow(x, 3) - 48 * Math.pow(x, 2) + 32 * x - 7;
    softLeakyAmeoY = -8 * (x * x * x * x) + 32 * (x * x * x) - 48 * (x * x) + 32 * x - 7;
  } else softLeakyAmeoY = leakyness * (x - 1) + 1;

  softLeakyAmeoY = (softLeakyAmeoY - 0.5) * 2;

  const y0Mix = factor;
  const y1Mix = 1 - factor;

  return y0Mix * ameoY + y1Mix * softLeakyAmeoY;
};

const nativeFusedInterpolatedAmeoGradImplInner = (
  factor: number,
  x: number | string,
  leakyness: number | string,
  dy: number | string
): number => {
  if (typeof x === 'string' || typeof leakyness === 'string' || typeof dy === 'string') {
    throw new Error('nativeFusedInterpolatedAmeoImpl only supports number types');
  }

  x *= 0.5;
  x -= 0.5;

  let ameoGradY: number;
  if ((x <= -1 && x > -2) || (x <= 1 && x > 0)) {
    ameoGradY = 1;
  } else if (x <= 0 && x > -1) {
    ameoGradY = -1;
  } else {
    ameoGradY = 0;
  }

  let softLeakyAmeoGradY: number;
  if (x <= -2) softLeakyAmeoGradY = leakyness;
  else if (x <= -1.5) {
    // softLeakyAmeoGradY = 32 * Math.pow(x + 2, 3);
    const xPlus2 = x + 2;
    softLeakyAmeoGradY = 32 * (xPlus2 * xPlus2 * xPlus2);
  } else if (x <= -0.5) {
    // softLeakyAmeoGradY = -32 * Math.pow(x, 3) - 96 * Math.pow(x, 2) - 96 * x - 32;
    softLeakyAmeoGradY = -32 * (x * x * x) - 96 * (x * x) - 96 * x - 32;
  } else if (x <= 0.5) {
    // softLeakyAmeoGradY = 32 * Math.pow(x, 3);
    softLeakyAmeoGradY = 32 * (x * x * x);
  } else if (x <= 1) {
    // softLeakyAmeoGradY = -32 * Math.pow(x, 3) + 96 * Math.pow(x, 2) - 96 * x + 32;
    softLeakyAmeoGradY = -32 * (x * x * x) + 96 * (x * x) - 96 * x + 32;
  } else softLeakyAmeoGradY = leakyness;

  const y0Mix = factor;
  const y1Mix = 1 - factor;

  return (y0Mix * ameoGradY + y1Mix * softLeakyAmeoGradY) * dy;
};

const nativeFusedInterpolatedAmeoImpl = (
  factor: number,
  leakyness: number,
  xsShape: number[],
  xs: TypedArray | string[],
  dtype: DataType
) => {
  const resultSize = tfc.util.sizeFromShape(xsShape);
  const result = tfc.util.getTypedArrayFromDType(dtype as any, resultSize);

  for (let i = 0; i < result.length; ++i) {
    result[i] = nativeFusedInterpolatedAmeoImplInner(factor, xs[i % xs.length], leakyness);
  }

  return [result, xsShape];
};

const nativeFusedInterpolatedAmeoGradImpl = (
  factor: number,
  leakyness: number,
  xsShape: number[],
  xs: TypedArray | string[],
  dyVals: TypedArray | string[],
  dtype: DataType
) => {
  const resultSize = tfc.util.sizeFromShape(xsShape);
  const result = tfc.util.getTypedArrayFromDType(dtype as any, resultSize);

  for (let i = 0; i < result.length; ++i) {
    result[i] = nativeFusedInterpolatedAmeoGradImplInner(
      factor,
      xs[i % xs.length],
      leakyness,
      dyVals[i % dyVals.length]
    );
  }

  return [result, xsShape];
};

tfc.registerKernel({
  kernelName: 'fusedInterpolatedAmeo',
  backendName: 'cpu',
  kernelFunc: ({
    inputs,
    backend,
  }: {
    inputs: NamedTensorInfoMap;
    backend: any;
    attrs?: NamedAttrMap;
  }) => {
    const { factor, leakyness, x } = inputs;
    if (!factor || !leakyness || !x) {
      throw new Error('fusedInterpolatedAmeo requires three inputs: factor, leakyness, x');
    }
    const cpuBackend = backend;
    const factorVals = cpuBackend.data.get(factor.dataId).values;
    const leakynessVals = cpuBackend.data.get(leakyness.dataId).values;
    const xVals = cpuBackend.data.get(x.dataId).values;
    const $dtype = x.dtype;

    if (wasmEngine) {
      const ys = wasmEngine.apply_batch_fused_interpolated_ameo(
        factorVals[0],
        leakynessVals[0],
        xVals
      );
      return backend.makeTensorInfo(x.shape, $dtype, ys);
    }

    const [resultData, resultShape] = nativeFusedInterpolatedAmeoImpl(
      factorVals[0],
      leakynessVals[0],
      x.shape,
      xVals,
      $dtype
    );
    return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
  },
});

tfc.registerKernel({
  kernelName: 'fusedInterpolatedAmeoGrad',
  backendName: 'cpu',
  kernelFunc: ({
    inputs,
    backend,
  }: {
    inputs: NamedTensorInfoMap;
    backend: any;
    attrs?: NamedAttrMap;
  }) => {
    const { factor, leakyness, x, dy } = inputs;
    if (!factor || !leakyness || !x || !dy) {
      throw new Error('fusedInterpolatedAmeoGrad requires three inputs: factor, leakyness, x');
    }
    const cpuBackend = backend;
    // assertNotComplex([a, b], name);
    const factorVals = cpuBackend.data.get(factor.dataId).values;
    const leakynessVals = cpuBackend.data.get(leakyness.dataId).values;
    const xVals = cpuBackend.data.get(x.dataId).values;
    const dyVals = cpuBackend.data.get(dy.dataId).values;
    const $dtype = x.dtype;

    if (wasmEngine) {
      const ys = wasmEngine.apply_batch_fused_interpolated_ameo_grad(
        factorVals[0],
        leakynessVals[0],
        xVals,
        dyVals
      );
      return backend.makeTensorInfo(x.shape, $dtype, ys);
    }

    const [resultData, resultShape] = nativeFusedInterpolatedAmeoGradImpl(
      factorVals[0],
      leakynessVals[0],
      x.shape,
      xVals,
      dyVals,
      $dtype
    );
    return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
  },
});

tfc.registerGradient({
  kernelName: 'fusedInterpolatedAmeo',
  inputsToSave: ['factor', 'leakyness', 'x'],
  gradFunc: (dy, saved) => {
    if (Array.isArray(dy)) {
      throw new Error('ameo gradient only supports a single tensor');
    }
    return {
      x: () =>
        engine().runKernel('fusedInterpolatedAmeoGrad', {
          factor: saved[0],
          leakyness: saved[1],
          x: saved[2],
          dy,
        }) as Tensor<Rank>,
    };
  },
});

/**
 * Interpolation between Ameo and SoftLeakyAmeo
 */
export class InterpolatedAmeo extends Activation {
  private factor: number;
  private leakyness: number | undefined;

  constructor(factor: number, leakyness?: number) {
    if (factor < 0 || factor > 1) {
      throw new Error('`factor` must be between 0 and 1');
    }
    super();
    this.factor = factor;
    this.leakyness = leakyness;
  }

  apply(tensor: Tensor<Rank>, _axis?: number | undefined): Tensor<Rank> {
    if (engine().backendName === 'cpu') {
      return engine().runKernel('fusedInterpolatedAmeo', {
        factor: tfc.scalar(this.factor),
        leakyness: tfc.scalar(this.leakyness ?? 0.05),
        x: tensor,
      });
    }

    const y0Mix = this.factor;
    const y1Mix = 1 - this.factor;

    const y0: Tensor<Rank> = (engine().runKernel('ameo', { x: tensor }) as Tensor<Rank>).mul(y0Mix);
    const y1: Tensor<Rank> = (
      engine().runKernel('softLeakyAmeo', {
        a: tensor,
        b: tfc.scalar(this.leakyness ?? 0.05),
      }) as Tensor<Rank>
    ).mul(y1Mix);
    return y0.add(y1);
  }
}
