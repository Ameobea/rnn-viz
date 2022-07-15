use rand::Rng;
use wasm_bindgen::prelude::*;

pub static mut RNG: *mut rand_pcg::Pcg64 = std::ptr::null_mut();

fn rng() -> &'static mut impl rand::Rng {
  unsafe { &mut *RNG }
}

const stream: u128 = 0x902bdbf7bb116d7ac28fa16a61abf96;

#[wasm_bindgen(start)]
pub fn main() {
  unsafe {
    RNG = Box::into_raw(Box::new(rand_pcg::Pcg64::new(0x1afef60dd15ea5e5, stream)));
  }
}

#[wasm_bindgen]
pub fn seed_rng(seed1: f64, seed2: f64) {
  unsafe {
    if !RNG.is_null() {
      drop(Box::from_raw(RNG));
    }
    RNG = Box::into_raw(Box::new(rand_pcg::Pcg64::new(
      std::mem::transmute((seed1, seed2)),
      stream,
    )));
  }
}

#[wasm_bindgen]
pub fn wrapping_unsigned_8_bit_add(batch_size: usize) -> Vec<f32> {
  let bits_per_u8 = 8;
  let mut vals = Vec::with_capacity(batch_size * bits_per_u8 * 3);

  let rng = rng();

  for _ in 0..batch_size {
    let a = rng.gen::<u8>();
    let b = rng.gen::<u8>();
    let y = a.wrapping_add(b);

    vals.extend(u8_to_bits(a));
    vals.extend(u8_to_bits(b));
    vals.extend(u8_to_bits(y));
  }

  vals
}

#[wasm_bindgen]
pub fn wrapping_unsigned_8_bit_add_full_validation() -> Vec<f32> {
  let bits_per_u8 = 8;
  let mut vals = Vec::with_capacity(256 * 256 * bits_per_u8 * 3);

  for a in 0..=255u8 {
    for b in 0..=255u8 {
      let y = a.wrapping_add(b);

      vals.extend(u8_to_bits(a));
      vals.extend(u8_to_bits(b));
      vals.extend(u8_to_bits(y));
    }
  }

  vals
}

fn u8_to_bits(val: u8) -> impl Iterator<Item = f32> {
  (0..8).map(move |i| if (val >> i) & 1 == 0 { -1. } else { 1. })
}

#[wasm_bindgen]
pub fn saturating_unsigned_8_bit_add(batch_size: usize) -> Vec<f32> {
  let bits_per_u8 = 8;
  let mut vals = Vec::with_capacity(batch_size * bits_per_u8 * 3);

  let rng = rng();

  for _ in 0..batch_size {
    let a = rng.gen::<u8>();
    let b = rng.gen::<u8>();
    let y = a.saturating_add(b);

    vals.extend(u8_to_bits(a));
    vals.extend(u8_to_bits(b));
    vals.extend(u8_to_bits(y));
  }

  vals
}

#[wasm_bindgen]
pub fn eight_bit_unsigned_binary_count(seq_len: usize) -> Vec<f32> {
  let mut vals = Vec::with_capacity(seq_len * 8);

  for i in 0..seq_len {
    vals.extend(u8_to_bits(i as u8));
  }

  vals
}
