use z3::ast::{Ast, Bool, Float, Real};

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

fn sum<'a>(ctx: &'a z3::Context, vals: &[Real<'a>]) -> Real<'a> {
    let mut sum = Real::from_real(ctx, 0, 1);
    println!("vals: {:?}", vals);
    for val in vals {
        sum = Real::add(ctx, &[&sum, &val]);
    }
    sum
}

fn bool_to_real<'a>(ctx: &'a z3::Context, val: bool) -> Real<'a> {
    Real::from_real(ctx, if val { 1 } else { -1 }, 1)
}

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
    // let numerators = 0..=64;
    let numerators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        .into_iter()
        .flat_map(|v| [v, v * 2]);
    // let denominators = 1..=48;
    let denominators = [1, 2, 3].into_iter();
    numerators.flat_map(move |n| {
        denominators
            .clone()
            .map(move |d| n as f32 / d as f32)
            .flat_map(|f| [f, -f])
    })
}

fn solve_brute_force(truth_table: &[((bool, bool, bool), bool)]) -> Option<(f32, f32, f32, f32)> {
    let mut valid_vals = valid_vals().collect::<Vec<_>>();
    valid_vals.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    valid_vals.dedup();

    for &z_weight in &valid_vals {
        for &y_weight in &valid_vals {
            for &x_weight in &valid_vals {
                'outer: for &bias in &valid_vals {
                    for ((x, y, z), expected) in truth_table {
                        let x = if *x { 1. } else { -1. };
                        let y = if *y { 1. } else { -1. };
                        let z = if *z { 1. } else { -1. };
                        let x = x_weight * x;
                        let y = y_weight * y;
                        let z = z_weight * z;
                        let expected = if *expected { 1. } else { -1. };
                        let actual = scaled_shifted_ameo_activation(x + y + z + bias);
                        if (actual - expected).abs() > 1e-6 {
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

fn ternary_or_all_false_or_all_but_one_true(x: bool, y: bool, z: bool) -> bool {
    (if x { y } else { z })
        || (!x && !y && !z)
        || (!x && y && z)
        || (x && !y && z)
        || (x && y && !z)
}

fn sanity() {
    let (x, y, z) = (-1., -1., -1.);
    let (x_weight, y_weight, z_weight, bias) = (-1. / 3., 1. / 3., 7. / 6., 5. / 6.);
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

fn main() {
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
        solve_brute_force(&truth_table).expect("No Solution");
    println!(
        "ternary: x_weight: {}, y_weight: {}, z_weight: {}, bias: {}",
        x_weight, y_weight, z_weight, bias
    );

    let truth_table = gen_3_input_truth_table(ternary_or_all_false_or_all_but_one_true);
    let (x_weight, y_weight, z_weight, bias) =
        solve_brute_force(&truth_table).expect("No Solution");
    println!(
        "ternary_or_all_false_or_all_but_one_true: x_weight: {}, y_weight: {}, z_weight: {}, bias: {}",
        x_weight, y_weight, z_weight, bias
    );

    let mut step_ix = 0;
    let mut no_solution_count = 0;
    for v_0 in [false, true] {
        for v_1 in [false, true] {
            for v_2 in [false, true] {
                for v_3 in [false, true] {
                    for v_4 in [false, true] {
                        for v_5 in [false, true] {
                            for v_6 in [false, true] {
                                for v_7 in [false, true] {
                                    let truth_table = vec![
                                        ((false, false, false), v_0),
                                        ((false, false, true), v_1),
                                        ((false, true, false), v_2),
                                        ((false, true, true), v_3),
                                        ((true, false, false), v_4),
                                        ((true, false, true), v_5),
                                        ((true, true, false), v_6),
                                        ((true, true, true), v_7),
                                    ];
                                    match solve_brute_force(&truth_table) {
                                        Some((x_weight, y_weight, z_weight, bias)) => {
                                            // println!(
                                            //     "[{step_ix}]: x_weight: {x_weight}, y_weight: {y_weight}, z_weight: {z_weight}, bias: {bias}",
                                            // )
                                        }
                                        None => {
                                            println!("[{step_ix}] No Solution: {:?}", truth_table);
                                            no_solution_count += 1;
                                        }
                                    };

                                    step_ix += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("No solution for {}/{} cases", no_solution_count, step_ix);
}
