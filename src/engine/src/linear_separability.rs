use wasm_bindgen::prelude::*;

use crate::batch_activation::{apply_ameo, apply_interpolated_ameo, apply_soft_leaky_ameo};

fn activation_from_id(id: usize) -> fn(f32) -> f32 {
  match id {
    // linear
    0 => |x: f32| x,
    // relu
    1 => |x: f32| if x < 0. { 0. } else { x },
    // sigmoid
    2 => |x: f32| 1. / (1. + (-x).exp()),
    // gcu
    3 => |x: f32| x * x.cos(),
    // tanh
    4 => |x: f32| x.tanh(),
    // ameo
    5 => apply_ameo,
    // soft leaky ameo
    6 => |x: f32| apply_soft_leaky_ameo(0.01, x),
    // interpolated ameo
    7 => |x: f32| apply_interpolated_ameo(0.1, 0.01, x),
    // gaussian
    8 => |x: f32| std::f32::consts::E.powf(-x * x),
    _ => panic!("activation_from_id: invalid id"),
  }
}

// const RANGE: [f32; 2] = [-1.2, 1.2];

#[wasm_bindgen]
pub fn plot_classification(
  size: usize,
  function_id: usize,
  x_weight: f32,
  y_weight: f32,
  bias: f32,
  positive_target_value: f32,
  negative_target_value: f32,
) -> Vec<u8> {
  let mut image_data = Vec::with_capacity(size * size * 4);
  let range = [negative_target_value, positive_target_value];

  // let positive_target_value = 1.;
  // let negative_target_value = -1.;
  let decision_boundary = (negative_target_value + positive_target_value) / 2.;
  let decision_range = positive_target_value - negative_target_value;

  for y_ix in 0..size {
    let y = range[0] + (range[1] - range[0]) * y_ix as f32 / (size - 1) as f32;
    for x_ix in 0..size {
      let x = range[0] + (range[1] - range[0]) * x_ix as f32 / (size - 1) as f32;

      let sum = x_weight * x + y_weight * y + bias;
      let activated = activation_from_id(function_id)(sum);
      let color = match activated {
        // decision boundary
        x if (decision_boundary - x).abs() < 0.02 => [0xff, 0xff, 0xff, 0xff],
        // perfect positive
        x if (positive_target_value - x).abs() < 0.02 => [0, 0xcc, 0, 0xff],
        // perfect negative
        x if (negative_target_value - x).abs() < 0.02 => [0xff, 0, 0, 0xff],
        // negative
        x if x < decision_boundary => {
          let diff = (x - decision_boundary).abs().min(decision_range / 2.) * 1.;
          [
            0x22u8.saturating_add((diff * (255. - 80.)) as u8),
            0,
            0,
            0xbf,
          ]
        }
        // positive
        x if x >= decision_boundary => {
          let diff = (x - decision_boundary).abs().min(decision_range / 2.) * 1.;
          [
            0,
            0x22u8.saturating_add((diff * (255. - 36.)) as u8),
            0,
            0xbf,
          ]
        }
        _ => unreachable!(),
      };
      image_data.extend_from_slice(&color);
    }
  }

  image_data
}
