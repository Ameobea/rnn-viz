#![feature(box_syntax)]

use std::collections::HashMap;

use itertools::Itertools;
use nanoserde::SerJson;
use rand::Rng;
use rand_distr::{Distribution, Normal};
#[cfg(feature = "z3-support")]
use z3::ast::{Ast, Bool, Float, Real};

use crate::{
    boolean::{binary_to_dec, compute_boolean_complexities, paper_repro},
    three_input_complexities::THREE_INPUT_BOOLEAN_COMPLEXITIES,
};

mod boolean;
mod three_input_complexities;

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
    let y = ameo_activation(0.5 * x - 0.5);
    (y - 0.5) * 2.
}

#[cfg(feature = "z3-support")]
fn between<'a>(
    ctx: &'a z3::Context,
    min_exclusive: Option<i32>,
    max_inclusive: Option<i32>,
    val: &Real<'a>,
) -> Bool<'a> {
    let cond1 = min_exclusive.map(|min_exclusive| val.gt(&Real::from_real(ctx, min_exclusive, 1)));
    let cond2 = max_inclusive.map(|max_inclusive| val.le(&Real::from_real(ctx, max_inclusive, 1)));

    match (cond1, cond2) {
        (Some(cond1), Some(cond2)) => Bool::and(ctx, &[&cond1, &cond2]),
        (Some(cond), None) | (None, Some(cond)) => cond,
        (None, None) => Bool::from_bool(ctx, true),
    }
}

#[cfg(feature = "z3-support")]
fn z3_ameo_activation<'a>(
    ctx: &'a z3::Context,
    x: Real<'a>,
    x_weight: &'a Real<'a>,
    y: Real<'a>,
    y_weight: &'a Real<'a>,
    z: Real<'a>,
    z_weight: &'a Real<'a>,
    bias: &'a Real<'a>,
) -> Real<'a> {
    let x_val = Real::mul(ctx, &[&x, &x_weight]);
    let y_val = Real::mul(ctx, &[&y, &y_weight]);
    let z_val = Real::mul(ctx, &[&z, &z_weight]);
    let sum = Real::add(ctx, &[&x_val, &y_val, &z_val, &bias]);

    // Pass sum to activation function
    let x = Real::mul(ctx, &[&sum, &Real::from_real(ctx, 15, 10)]);
    let x = Real::sub(ctx, &[&x, &Real::from_real(ctx, 1, 2)]);

    let part_1_guard = between(ctx, None, Some(-2), &x);
    let part_1 = Real::from_real(ctx, 0, 1);
    let part_2_guard = between(ctx, Some(-1), Some(0), &x);
    let part_2 = Real::add(ctx, &[&x, &Real::from_real(ctx, 2, 1)]);
    let part_3_guard = between(ctx, Some(0), Some(1), &x);
    let part_3 = Real::add(ctx, &[&x, &Real::from_real(ctx, 0, 1)]);
    let part_4_guard = between(ctx, Some(1), None, &x);
    let part_4 = Real::from_real(ctx, 1, 1);

    let part_1 = part_1_guard.ite(&part_1, &Real::from_real(ctx, 0, 1));
    let part_2 = part_2_guard.ite(&part_2, &Real::from_real(ctx, 0, 1));
    let part_3 = part_3_guard.ite(&part_3, &Real::from_real(ctx, 0, 1));
    let part_4 = part_4_guard.ite(&part_4, &Real::from_real(ctx, 0, 1));

    let activation = Real::add(ctx, &[&part_1, &part_2, &part_3, &part_4]);

    let activation = Real::sub(ctx, &[&activation, &Real::from_real(ctx, 1, 2)]);
    Real::mul(ctx, &[&activation, &Real::from_real(ctx, 2, 1)])
}

fn gen_3_input_truth_table(
    function: impl Fn(bool, bool, bool) -> bool,
) -> Vec<((bool, bool, bool), bool)> {
    let mut truth_table = vec![];
    for x in [false, true].iter() {
        for y in [false, true].iter() {
            for z in [false, true].iter() {
                truth_table.push(((*x, *y, *z), function(*x, *y, *z)));
            }
        }
    }
    truth_table
}

#[cfg(feature = "z3-support")]
fn sum<'a>(ctx: &'a z3::Context, vals: &[Real<'a>]) -> Real<'a> {
    let mut sum = Real::from_real(ctx, 0, 1);
    println!("vals: {:?}", vals);
    for val in vals {
        sum = Real::add(ctx, &[&sum, &val]);
    }
    sum
}

#[cfg(feature = "z3-support")]
fn bool_to_real<'a>(ctx: &'a z3::Context, val: bool) -> Real<'a> {
    Real::from_real(ctx, if val { 1 } else { -1 }, 1)
}

#[cfg(feature = "z3-support")]
fn solve_z3() {
    let z3_conf = z3::Config::new();
    let ctx = z3::Context::new(&z3_conf);
    let optimizer = z3::Optimize::new(&ctx);

    let x_weight = Real::new_const(&ctx, "x_weight");
    let y_weight = Real::new_const(&ctx, "y_weight");
    let z_weight = Real::new_const(&ctx, "z_weight");
    let bias = Real::new_const(&ctx, "bias");

    let ternary_conditional = |x: bool, y: bool, z: bool| -> bool {
        if x {
            y
        } else {
            z
        }
    };

    let truth_table = gen_3_input_truth_table(ternary_conditional);

    let mut losses = Vec::new();
    for ((x, y, z), expected) in &truth_table {
        let x = bool_to_real(&ctx, *x);
        let y = bool_to_real(&ctx, *y);
        let z = bool_to_real(&ctx, *z);
        let expected = bool_to_real(&ctx, *expected);
        let actual = z3_ameo_activation(&ctx, x, &x_weight, y, &y_weight, z, &z_weight, &bias);
        let loss = Real::sub(&ctx, &[&actual, &expected]);
        let loss = Real::mul(&ctx, &[&loss, &loss]);
        losses.push(loss);
    }
    let ast = sum(&ctx, &losses);

    // // sanity check
    // let model = optimizer.get_model().unwrap();
    // let x_weight = Real::from_real(&ctx, -4, 10);
    // let y_weight = Real::from_real(&ctx, 3477, 10_000);
    // let z_weight = Real::from_real(&ctx, 11689, 10_000);
    // let bias = Real::from_real(&ctx, 8875, 10_000);

    // for ((x, y, z), expected) in truth_table {
    //     let x = bool_to_real(&ctx, *x);
    //     let y = bool_to_real(&ctx, *y);
    //     let z = bool_to_real(&ctx, *z);
    //     let expected = bool_to_real(&ctx, *expected);
    //     let actual = z3_ameo_activation(&ctx, x, &x_weight, y, &y_weight, z, &z_weight, &bias);
    //     println!("expected: {:?}, actual: {:?}", expected,actual);
    // }

    optimizer.minimize(&ast);

    println!("Starting solving...");
    let res = optimizer.check(&[]);
    println!("Sat result: {:?}", res);
    let model = optimizer.get_model().unwrap();

    println!(
        "x_weight: {}, y_weight: {}, z_weight: {}, bias: {}",
        model.eval(&x_weight, true).unwrap(),
        model.eval(&y_weight, true).unwrap(),
        model.eval(&z_weight, true).unwrap(),
        model.eval(&bias, true).unwrap()
    );
}

fn valid_vals() -> impl Iterator<Item = f32> {
    // // let numerators = 0..=64;
    // let numerators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    //     .into_iter()
    //     .flat_map(|v| [v, v * 2]);
    // // let denominators = 1..=48;
    // let denominators = [1, 2, 3].into_iter();
    // numerators.flat_map(move |n| {
    //     denominators
    //         .clone()
    //         .map(move |d| n as f32 / d as f32)
    //         .flat_map(|f| [f, -f])
    // })
    [
        -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.,
        4.5,
    ]
    .into_iter()
}

fn solve_brute_force(
    truth_table: &[((bool, bool, bool), bool)],
    accept_imperfect: bool,
) -> Option<(f32, f32, f32, f32)> {
    let mut valid_vals = valid_vals().collect::<Vec<_>>();
    valid_vals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    valid_vals.dedup();
    // println!("valid vals: {:?}", valid_vals);

    for &z_weight in &valid_vals {
        for &y_weight in &valid_vals {
            for &x_weight in &valid_vals {
                'outer: for &bias in &valid_vals {
                    for ((x, y, z), expected) in truth_table {
                        let x = if *x { 1. } else { -1. };
                        let y = if *y { 1. } else { -1. };
                        let z = if *z { 1. } else { -1. };
                        let x_in = x_weight * x;
                        let y_in = y_weight * y;
                        let z_in = z_weight * z;
                        let expected: f32 = if *expected { 1. } else { -1. };
                        let actual = scaled_shifted_ameo_activation(x_in + y_in + z_in + bias);

                        if accept_imperfect {
                            if actual == 0. || actual.signum() != expected.signum() {
                                continue 'outer;
                            }
                        } else if (actual - expected).abs() > 1e-6 {
                            continue 'outer;
                        }
                    }
                    return Some((x_weight, y_weight, z_weight, bias));
                }
            }
        }
    }

    None
}

fn solve_brute_force_4(
    truth_table: &[((bool, bool, bool, bool), bool)],
    accept_imperfect: bool,
) -> Option<(f32, f32, f32, f32, f32)> {
    let mut valid_vals = valid_vals().collect::<Vec<_>>();
    valid_vals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    valid_vals.dedup();
    // println!("valid vals: {:?}", valid_vals);

    for &w_weight in &valid_vals {
        for &z_weight in &valid_vals {
            for &y_weight in &valid_vals {
                for &x_weight in &valid_vals {
                    'outer: for &bias in &valid_vals {
                        for ((x, y, z, w), expected) in truth_table {
                            let x = if *x { 1. } else { -1. };
                            let y = if *y { 1. } else { -1. };
                            let z = if *z { 1. } else { -1. };
                            let w = if *w { 1. } else { -1. };
                            let x_in = x_weight * x;
                            let y_in = y_weight * y;
                            let z_in = z_weight * z;
                            let w_in = w_weight * w;
                            let expected: f32 = if *expected { 1. } else { -1. };
                            let actual =
                                scaled_shifted_ameo_activation(x_in + y_in + z_in + w_in + bias);

                            if accept_imperfect {
                                if actual == 0. || actual.signum() != expected.signum() {
                                    continue 'outer;
                                }
                            } else if (actual - expected).abs() > 1e-6 {
                                continue 'outer;
                            }
                        }
                        return Some((x_weight, y_weight, z_weight, w_weight, bias));
                    }
                }
            }
        }
    }

    None
}

fn ternary_or_all_false_or_all_but_one_true(x: bool, y: bool, z: bool) -> bool {
    (if x { y } else { z })
        || (!x && !y && !z)
        || (!x && y && z)
        || (x && !y && z)
        || (x && y && !z)
}

fn sanity() {
    let (x, y, z) = (-1., -1., -1.);
    let (x_weight, y_weight, z_weight, bias) = (-1., -1., 2., 3.);
    let res = scaled_shifted_ameo_activation(x * x_weight + y * y_weight + z * z_weight + bias);
    assert_eq!(res, 1.);

    for z in [false, true] {
        for y in [false, true] {
            for x in [false, true] {
                let actual = scaled_shifted_ameo_activation(
                    if x { 1. } else { -1. } * x_weight
                        + if y { 1. } else { -1. } * y_weight
                        + if z { 1. } else { -1. } * z_weight
                        + bias,
                );
                let expected = if ternary_or_all_false_or_all_but_one_true(x, y, z) {
                    1.
                } else {
                    -1.
                };
                assert!((expected - actual).abs() < 1e-6);
            }
        }
    }
}

fn sanity2() {
    let truth_table = [
        ((false, false, false), false),
        ((false, false, true), false),
        ((false, true, false), false),
        ((false, true, true), true),
        ((true, false, false), true),
        ((true, false, true), false),
        ((true, true, false), false),
        ((true, true, true), false),
    ];

    let sol = solve_brute_force(&truth_table, true).expect("no solution");
    println!("sol: {:?}", sol);
}

fn build_all_3_input_truth_tables() -> Vec<[((bool, bool, bool), bool); 8]> {
    let mut tables = Vec::new();

    for v_0 in [false, true] {
        for v_1 in [false, true] {
            for v_2 in [false, true] {
                for v_3 in [false, true] {
                    for v_4 in [false, true] {
                        for v_5 in [false, true] {
                            for v_6 in [false, true] {
                                for v_7 in [false, true] {
                                    let truth_table = [
                                        ((false, false, false), v_0),
                                        ((false, false, true), v_1),
                                        ((false, true, false), v_2),
                                        ((false, true, true), v_3),
                                        ((true, false, false), v_4),
                                        ((true, false, true), v_5),
                                        ((true, true, false), v_6),
                                        ((true, true, true), v_7),
                                    ];
                                    tables.push(truth_table);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    tables
}

fn build_all_4_input_truth_tables() -> Vec<[((bool, bool, bool, bool), bool); 16]> {
    let mut tables = Vec::new();

    for v_0 in [false, true] {
        for v_1 in [false, true] {
            for v_2 in [false, true] {
                for v_3 in [false, true] {
                    for v_4 in [false, true] {
                        for v_5 in [false, true] {
                            for v_6 in [false, true] {
                                for v_7 in [false, true] {
                                    for v_8 in [false, true] {
                                        for v_9 in [false, true] {
                                            for v_10 in [false, true] {
                                                for v_11 in [false, true] {
                                                    for v_12 in [false, true] {
                                                        for v_13 in [false, true] {
                                                            for v_14 in [false, true] {
                                                                for v_15 in [false, true] {
                                                                    let truth_table = [
                                                                        (
                                                                            (
                                                                                false, false,
                                                                                false, false,
                                                                            ),
                                                                            v_0,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, false,
                                                                                false, true,
                                                                            ),
                                                                            v_1,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, false, true,
                                                                                false,
                                                                            ),
                                                                            v_2,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, false, true,
                                                                                true,
                                                                            ),
                                                                            v_3,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, true, false,
                                                                                false,
                                                                            ),
                                                                            v_4,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, true, false,
                                                                                true,
                                                                            ),
                                                                            v_5,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, true, true,
                                                                                false,
                                                                            ),
                                                                            v_6,
                                                                        ),
                                                                        (
                                                                            (
                                                                                false, true, true,
                                                                                true,
                                                                            ),
                                                                            v_7,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, false, false,
                                                                                false,
                                                                            ),
                                                                            v_8,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, false, false,
                                                                                true,
                                                                            ),
                                                                            v_9,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, false, true,
                                                                                false,
                                                                            ),
                                                                            v_10,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, false, true,
                                                                                true,
                                                                            ),
                                                                            v_11,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, true, false,
                                                                                false,
                                                                            ),
                                                                            v_12,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, true, false,
                                                                                true,
                                                                            ),
                                                                            v_13,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, true, true,
                                                                                false,
                                                                            ),
                                                                            v_14,
                                                                        ),
                                                                        (
                                                                            (
                                                                                true, true, true,
                                                                                true,
                                                                            ),
                                                                            v_15,
                                                                        ),
                                                                    ];
                                                                    tables.push(truth_table);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    tables
}

fn fmt_bool(b: bool) -> &'static str {
    if b {
        "T"
    } else {
        "F"
    }
}

fn solve_all_3_input_truth_tables() {
    let mut step_ix = 0;
    let mut no_solution_count = 0;
    let mut imperfect_solution_count = 0;
    for truth_table in build_all_3_input_truth_tables() {
        if let Some((x_weight, y_weight, z_weight, bias)) = solve_brute_force(&truth_table, false) {
            println!("[{step_ix}]: x_weight: {x_weight}, y_weight: {y_weight}, z_weight: {z_weight}, bias: {bias}");
            step_ix += 1;
            continue;
        }

        // No perfect solution with integer-valued weights; try to find linearly separable
        match solve_brute_force(&truth_table, true) {
            Some((x_weight, y_weight, z_weight, bias)) => {
                imperfect_solution_count += 1;
                println!("[{step_ix}] {{NON_PERFECT}}: x_weight: {x_weight}, y_weight: {y_weight}, z_weight: {z_weight}, bias: {bias}");
                println!(
                    "  {:?}",
                    truth_table
                        .into_iter()
                        .map(|(i, o)| format!(
                            "{}{}{}: {}",
                            fmt_bool(i.0),
                            fmt_bool(i.1),
                            fmt_bool(i.2),
                            fmt_bool(o)
                        ))
                        .collect::<Vec<_>>()
                );
            }
            None => {
                println!(
                    "[{step_ix}] No Solution: {:?}",
                    truth_table
                        .into_iter()
                        .map(|(i, o)| format!(
                            "{}{}{}: {}",
                            fmt_bool(i.0),
                            fmt_bool(i.1),
                            fmt_bool(i.2),
                            fmt_bool(o)
                        ))
                        .collect::<Vec<_>>()
                );
                no_solution_count += 1;
            }
        }

        step_ix += 1;
    }

    println!("\nNo solution for {}/{} cases", no_solution_count, step_ix);
    println!(
        "Imperfect solution for {}/{} cases",
        imperfect_solution_count, step_ix
    );
    println!(
        "Perfect solution for {}/{} cases",
        step_ix - no_solution_count - imperfect_solution_count,
        step_ix
    );
}

fn solve_all_4_input_truth_tables() {
    let mut step_ix = 0;
    let mut no_solution_count = 0;
    let mut imperfect_solution_count = 0;
    for truth_table in build_all_4_input_truth_tables() {
        if let Some((x_weight, y_weight, z_weight, w_weight, bias)) =
            solve_brute_force_4(&truth_table, false)
        {
            println!("[{step_ix}]: x_weight: {x_weight}, y_weight: {y_weight}, z_weight: {z_weight}, w_weight: {w_weight}, bias: {bias}");
            step_ix += 1;
            continue;
        }

        // No perfect solution with integer-valued weights; try to find linearly separable
        match solve_brute_force_4(&truth_table, true) {
            Some((x_weight, y_weight, z_weight, w_weight, bias)) => {
                imperfect_solution_count += 1;
                println!("[{step_ix}] {{NON_PERFECT}}: x_weight: {x_weight}, y_weight: {y_weight}, z_weight: {z_weight}, w_weight: {w_weight}, bias: {bias}");
                println!(
                    "  {:?}",
                    truth_table
                        .into_iter()
                        .map(|(i, o)| format!(
                            "{}{}{}{}: {}",
                            fmt_bool(i.0),
                            fmt_bool(i.1),
                            fmt_bool(i.2),
                            fmt_bool(i.3),
                            fmt_bool(o)
                        ))
                        .collect::<Vec<_>>()
                );
            }
            None => {
                println!(
                    "[{step_ix}] No Solution: {:?}",
                    truth_table
                        .into_iter()
                        .map(|(i, o)| format!(
                            "{}{}{}{}: {}",
                            fmt_bool(i.0),
                            fmt_bool(i.1),
                            fmt_bool(i.2),
                            fmt_bool(i.3),
                            fmt_bool(o)
                        ))
                        .collect::<Vec<_>>()
                );
                no_solution_count += 1;
            }
        }

        step_ix += 1;
    }

    println!("\nNo solution for {}/{} cases", no_solution_count, step_ix);
    println!(
        "Imperfect solution for {}/{} cases",
        imperfect_solution_count, step_ix
    );
    println!(
        "Perfect solution for {}/{} cases",
        step_ix - no_solution_count - imperfect_solution_count,
        step_ix
    );
}

/// Works backwards, figuring out which truth table is represented by all integer-valued parameters in a range.
fn solve_all_3_input_truth_tables_reverse() {
    let param_range = -5..=5;

    let mut counts_by_truth_table: HashMap<[((bool, bool, bool), bool); 8], usize> = HashMap::new();

    let mut total_attempts = 0usize;
    let mut no_solution_count = 0usize;
    for x_weight in param_range.clone() {
        for y_weight in param_range.clone() {
            for z_weight in param_range.clone() {
                'outer: for bias in param_range.clone() {
                    total_attempts += 1;

                    let mut truth_table = [
                        ((false, false, false), false),
                        ((false, false, true), false),
                        ((false, true, false), false),
                        ((false, true, true), false),
                        ((true, false, false), false),
                        ((true, false, true), false),
                        ((true, true, false), false),
                        ((true, true, true), false),
                    ];

                    let mut step_ix = 0;
                    for x in [false, true] {
                        for y in [false, true] {
                            for z in [false, true] {
                                let x = if x { 1. } else { -1. } * x_weight as f32;
                                let y = if y { 1. } else { -1. } * y_weight as f32;
                                let z = if z { 1. } else { -1. } * z_weight as f32;

                                let actual =
                                    scaled_shifted_ameo_activation(x + y + z + bias as f32);
                                let actual = if (actual - 1.).abs() < 1e-6 {
                                    true
                                } else if (actual + 1.).abs() < 1e-6 {
                                    false
                                } else {
                                    no_solution_count += 1;
                                    continue 'outer;
                                };
                                truth_table[step_ix].1 = actual;
                                step_ix += 1;
                            }
                        }
                    }

                    counts_by_truth_table
                        .entry(truth_table)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }
            }
        }
    }

    let mut truth_tables_by_count = HashMap::new();
    for (truth_table, count) in &counts_by_truth_table {
        truth_tables_by_count
            .entry(count)
            .and_modify(|truth_tables: &mut Vec<[((bool, bool, bool), bool); 8]>| {
                truth_tables.push(*truth_table)
            })
            .or_insert(vec![*truth_table]);
    }

    let mut truth_tables_by_count_sorted = truth_tables_by_count
        .into_iter()
        .map(|(count, truth_tables)| (count, truth_tables.len()))
        .collect::<Vec<_>>();
    truth_tables_by_count_sorted.sort_by(|a, b| b.0.cmp(&a.0));

    println!("\n[solution count]: [truth table count]");
    for (count, truth_tables_count) in truth_tables_by_count_sorted {
        println!("{}: {}", count, truth_tables_count);
    }

    println!(
        "\n\nTotal unique truth tables: {}",
        counts_by_truth_table.len()
    );

    println!(
        "Truth tables with no solution: {}",
        build_all_3_input_truth_tables().len() - counts_by_truth_table.len()
    );

    println!(
        "No valid binary truth table represented for {}/{} cases\n",
        no_solution_count, total_attempts
    );
}

fn binary_boolean_function_distributions(iters: usize) -> Vec<usize> {
    let mut counts_by_output: HashMap<[bool; 8], usize> = HashMap::new();

    let distr = Normal::new(0., 5.).unwrap();
    let init_weight = || distr.sample(&mut rand::thread_rng()) as f32;
    // let init_weight = || rand::thread_rng().gen_range((-5.)..5.) as f32;

    for _ in 0..iters {
        let x_weight = init_weight();
        let y_weight = init_weight();
        let z_weight = init_weight();
        let bias = init_weight();

        let mut i = 0usize;
        let mut outputs = [false; 8];
        for z in [-1., 1.] {
            for y in [-1., 1.] {
                for x in [-1., 1.] {
                    let output = scaled_shifted_ameo_activation(
                        x * x_weight + y * y_weight + z * z_weight + bias,
                    );
                    outputs[i] = if output > 0. { true } else { false };
                    i += 1;
                }
            }
        }

        counts_by_output
            .entry(outputs)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    let mut counts_by_output_sorted = counts_by_output
        .iter()
        .map(|(outputs, count)| (outputs, count))
        .collect::<Vec<_>>();
    counts_by_output_sorted.sort_unstable_by_key(|(_outputs, count)| *count);

    println!("\n[output]: [count]");
    for (output, count) in &counts_by_output_sorted {
        println!("{:?}: {}", output, count);
    }

    let mut out = counts_by_output.into_iter().collect_vec();
    out.sort_unstable_by_key(|(outputs, _count)| binary_to_dec(outputs));
    out.into_iter().map(|(_outputs, count)| count).collect()
}

fn convert_formula(formula: &str) -> String {
    formula.replace("*", " & ").replace("+", " | ")
}

fn compare_complexities_to_distribution() {
    let total_distr_samples = 100_000_000;
    let distrs = binary_boolean_function_distributions(total_distr_samples);
    // let complexities = compute_boolean_complexities(3);
    let complexities = THREE_INPUT_BOOLEAN_COMPLEXITIES;

    assert_eq!(distrs.len(), complexities.len());

    let inputs = (0..3)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    let output_count = inputs.len();
    // all possible permutations of [bool; output_count]
    let outputs = (0..output_count)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    assert_eq!(outputs.len(), complexities.len());

    let formulas = include_str!("./3_input_bool_minimal.txt")
        .split("\n")
        .filter(|s| !s.is_empty())
        .collect_vec();
    assert_eq!(formulas.len(), outputs.len());

    #[derive(SerJson)]
    struct FunctionStats {
        pub number: u64,
        pub complexity: u8,
        pub sample_count: usize,
        pub area_fraction: f64,
        pub truth_table: Vec<bool>,
        pub formula: String,
    }

    let mut all_stats = Vec::new();

    println!("\n\n\n");
    for i in 0..outputs.len() {
        all_stats.push(FunctionStats {
            number: binary_to_dec(&outputs[i]),
            complexity: complexities[i],
            sample_count: distrs[i],
            area_fraction: distrs[i] as f64 / total_distr_samples as f64,
            truth_table: outputs[i].to_vec(),
            formula: convert_formula(formulas[i]),
        });
    }

    println!("{}", SerJson::serialize_json(&all_stats));
}

fn main() {
    sanity2();
    sanity();

    let ternary_conditional = |x: bool, y: bool, z: bool| -> bool {
        if x {
            y
        } else {
            z
        }
    };
    let truth_table = gen_3_input_truth_table(ternary_conditional);
    let (x_weight, y_weight, z_weight, bias) =
        solve_brute_force(&truth_table, false).expect("No Solution");
    println!(
        "ternary: x_weight: {}, y_weight: {}, z_weight: {}, bias: {}",
        x_weight, y_weight, z_weight, bias
    );

    let truth_table = gen_3_input_truth_table(ternary_or_all_false_or_all_but_one_true);
    let (x_weight, y_weight, z_weight, bias) =
        solve_brute_force(&truth_table, false).expect("No Solution");
    println!(
        "ternary_or_all_false_or_all_but_one_true: x_weight: {}, y_weight: {}, z_weight: {}, bias: {}",
        x_weight, y_weight, z_weight, bias
    );

    // solve_all_3_input_truth_tables();
    // solve_all_4_input_truth_tables();
    // solve_all_3_input_truth_tables_reverse();

    // paper_repro();

    compare_complexities_to_distribution();
}
