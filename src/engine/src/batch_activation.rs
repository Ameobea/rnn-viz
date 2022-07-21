use wasm_bindgen::prelude::*;

pub(crate) fn apply_soft_leaky_ameo(leakyness: f32, mut x: f32) -> f32 {
  x *= 0.5;
  x -= 0.5;

  let y = if x <= -2. {
    leakyness * (x + 2.)
  } else if x <= -1.5 {
    8. * (x + 2.).powi(4)
  } else if x <= -0.5 {
    -8. * x.powi(4) + -32. * x.powi(3) - 48. * x.powi(2) + -32. * x - 7.
  } else if x <= 0.5 {
    8. * x.powi(4)
  } else if x <= 1. {
    -8. * x.powi(4) + 32. * x.powi(3) - 48. * x.powi(2) + 32. * x - 7.
  } else {
    leakyness * (x - 1.) + 1.
  };

  (y - 0.5) * 2.
}

#[wasm_bindgen]
pub fn apply_batch_soft_leaky_ameo(leakyness: f32, xs: &[f32]) -> Vec<f32> {
  let mut out = Vec::with_capacity(xs.len());
  for x in xs.iter() {
    out.push(apply_soft_leaky_ameo(leakyness, *x));
  }
  out
}

pub fn apply_soft_leaky_ameo_grad(leakyness: f32, mut x: f32) -> f32 {
  x *= 0.5;
  x -= 0.5;

  if x <= -2. {
    leakyness
  } else if x <= -1.5 {
    32. * (x + 2.).powi(3)
  } else if x <= -0.5 {
    -32. * x.powi(3) - 96. * x.powi(2) - 96. * x - 32.
  } else if x <= 0.5 {
    32. * x.powi(3)
  } else if x <= 1. {
    -32. * x.powi(3) + 96. * x.powi(2) - 96. * x + 32.
  } else {
    leakyness
  }
}

#[wasm_bindgen]
pub fn apply_batch_soft_leaky_ameo_grad(leakyness: f32, xs: &[f32]) -> Vec<f32> {
  let mut out = Vec::with_capacity(xs.len());
  for x in xs.iter() {
    out.push(apply_soft_leaky_ameo_grad(leakyness, *x));
  }
  out
}

pub(crate) fn apply_ameo(mut x: f32) -> f32 {
  x *= 0.5;
  x -= 0.5;

  let y = if x <= -1. {
    (x + 2.).max(0.)
  } else if x <= 0. {
    -x
  } else {
    x.min(1.)
  };
  (y - 0.5) * 2.
}

fn apply_ameo_grad(mut x: f32) -> f32 {
  x *= 0.5;
  x -= 0.5;

  if (x <= -1. && x > -2.) || (x <= 1. && x > 0.) {
    1.
  } else if x <= 0. {
    -1.
  } else {
    0.
  }
}

#[wasm_bindgen]
pub fn apply_batch_fused_interpolated_ameo(factor: f32, leakyness: f32, xs: &[f32]) -> Vec<f32> {
  let xmix = factor;
  let ymix = 1. - xmix;

  let mut out = Vec::with_capacity(xs.len());
  for x in xs.iter() {
    let ameo_y = apply_ameo(*x);
    let soft_leaky_ameo_y = apply_soft_leaky_ameo(leakyness, *x);
    out.push(xmix * ameo_y + ymix * soft_leaky_ameo_y);
  }
  out
}

#[wasm_bindgen]
pub fn apply_batch_fused_interpolated_ameo_grad(
  factor: f32,
  leakyness: f32,
  xs: &[f32],
  dys: &[f32],
) -> Vec<f32> {
  let xmix = factor;
  let ymix = 1. - xmix;

  let mut out = Vec::with_capacity(xs.len());
  for (i, x) in xs.iter().enumerate() {
    let ameo_grad_y = apply_ameo_grad(*x);
    let soft_leaky_ameo_grad_y = apply_soft_leaky_ameo_grad(leakyness, *x);
    out.push((xmix * ameo_grad_y + ymix * soft_leaky_ameo_grad_y) * dys[i]);
  }
  out
}
