export const nativeFusedInterpolatedAmeoImplInner = (
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
