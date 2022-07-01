use wasm_bindgen::prelude::*;

const DOMAIN: [f32; 2] = [-1.0, 1.0];

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

fn ameo_activation(x: f32) -> f32 {
    if x <= -2. {
        0.
    } else if x <= -1. {
        x + 2.
    } else if x <= 0. {
        -x
    } else if x <= 1. {
        x
    } else {
        1.
    }
}

fn scaled_shifted_ameo_activation(x: f32) -> f32 {
    let y = ameo_activation(1.5 * x - 0.5);
    (y - 0.5) * 2.
}

// if (val <= -2)           return leakyness * (val + 2);
// else if (val <= -1.5)    return 8 * Math.pow(val + 2, 4);
// // else if (val <= -1)   return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
// else if (val <= -0.5)    return -8 * Math.pow(val, 4) + -32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + -32 * val - 7;
// else if (val <= 0.5)     return 8 * Math.pow(val, 4);
// else if (val <= 1)       return -8 * Math.pow(val, 4) + 32 * Math.pow(val, 3) + -48 * Math.pow(val, 2) + 32 * val - 7;
// else                     return leakyness * (val - 1) + 1;
fn soft_leaky_ameo(x: f32) -> f32 {
    let leakyness = 0.05;

    if x <= -2. {
        leakyness * (x + 2.)
    } else if x <= -1.5 {
        8. * (x + 2.).powi(4)
    } else if x <= -0.5 {
        -8. * x.powi(4) + -32. * x.powi(3) + -48. * x.powi(2) + -32. * x - 7.
    } else if x <= 0.5 {
        8. * x.powi(4)
    } else if x <= 1. {
        -8. * x.powi(4) + 32. * x.powi(3) + -48. * x.powi(2) + 32. * x - 7.
    } else {
        leakyness * (x - 1.) + 1.
    }
}

fn scaled_shifted_soft_leaky_ameo_activation(x: f32) -> f32 {
    let y = soft_leaky_ameo(1.5 * x - 0.5);
    (y - 0.5) * 2.
}

fn gcu(x: f32) -> f32 {
    x * x.cos()
}

static mut VOXEL_COMPUTE_SCRATCH: *mut Vec<bool> = std::ptr::null_mut();

// Could do a lot better with bitflags for this
fn get_voxel_compute_scratch(resolution: usize) -> &'static mut [bool] {
    unsafe {
        if VOXEL_COMPUTE_SCRATCH.is_null() {
            VOXEL_COMPUTE_SCRATCH =
                Box::into_raw(Box::new(vec![false; resolution * resolution * resolution]));
            return &mut *VOXEL_COMPUTE_SCRATCH;
        }

        if (*VOXEL_COMPUTE_SCRATCH).len() != resolution * resolution * resolution {
            (*VOXEL_COMPUTE_SCRATCH).resize(resolution * resolution * resolution, false);
        }
        let scratch = &mut *VOXEL_COMPUTE_SCRATCH;
        scratch.fill(false);
        scratch
    }
}

#[wasm_bindgen]
pub fn compute_voxel_positions(
    weight_x: f32,
    weight_y: f32,
    weight_z: f32,
    bias: f32,
    resolution: usize,
) -> Vec<f32> {
    // let mut voxel_positions = Vec::new();
    let voxel_size = (DOMAIN[1] - DOMAIN[0]) / resolution as f32;

    let scratch = get_voxel_compute_scratch(resolution);

    for i_z in 0..resolution {
        let z = DOMAIN[0] + i_z as f32 * voxel_size;
        for i_y in 0..resolution {
            let y = DOMAIN[0] + i_y as f32 * voxel_size;
            for i_x in 0..resolution {
                let x = DOMAIN[0] + i_x as f32 * voxel_size;
                // let activation_fn = scaled_shifted_ameo_activation;
                // let activation_fn = scaled_shifted_soft_leaky_ameo_activation;
                let activation_fn = gcu;
                let value = activation_fn(weight_x * x + weight_y * y + weight_z * z + bias);
                if value > 0.95 {
                    scratch[i_z * resolution * resolution + i_y * resolution + i_x] = true;
                }
            }
        }
    }

    let mut voxel_positions = Vec::new();
    let mut culled_count = 0;
    for i_z in 0..resolution {
        let z = DOMAIN[0] + i_z as f32 * voxel_size;
        for i_y in 0..resolution {
            let y = DOMAIN[0] + i_y as f32 * voxel_size;
            for i_x in 0..resolution {
                if !scratch[i_z * resolution * resolution + i_y * resolution + i_x] {
                    continue;
                }

                if i_x == 0
                    || i_x == resolution - 1
                    || i_y == 0
                    || i_y == resolution - 1
                    || i_z == 0
                    || i_z == resolution - 1
                {
                    // always include on voxels on the edges
                } else {
                    // if all 6 adjacent voxels are on, we can cull this one for performance
                    if scratch[(i_z - 1) * resolution * resolution + i_y * resolution + i_x]
                        && scratch[(i_z + 1) * resolution * resolution + i_y * resolution + i_x]
                        && scratch[i_z * resolution * resolution + (i_y - 1) * resolution + i_x]
                        && scratch[i_z * resolution * resolution + (i_y + 1) * resolution + i_x]
                        && scratch[i_z * resolution * resolution + i_y * resolution + (i_x - 1)]
                        && scratch[i_z * resolution * resolution + i_y * resolution + (i_x + 1)]
                    {
                        culled_count += 1;
                        continue;
                    }
                }

                let x = DOMAIN[0] + i_x as f32 * voxel_size;
                voxel_positions.push(x);
                voxel_positions.push(y);
                voxel_positions.push(z);
            }
        }
    }

    log(&format!("Culled: {}", culled_count));
    log(&format!("Rendered: {}", voxel_positions.len() / 3));

    voxel_positions
}
