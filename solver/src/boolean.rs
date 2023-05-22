use boolean_expression::{Expr, BDD};
use itertools::Itertools;

const CHARS: &[char] = &['a', 'b', 'c', 'd', 'e', 'f', 'g'];

fn build_expr(inputs: &[Vec<bool>], output: &[bool]) -> Expr<char> {
    let mut expr = Expr::Const(false);

    assert_eq!(inputs.len(), output.len());
    for (i, input) in inputs.iter().enumerate() {
        let output = output[i];
        if !output {
            continue;
        }

        let sub_expr = input
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if *v {
                    Expr::Terminal(CHARS[i])
                } else {
                    Expr::Not(Box::new(Expr::Terminal(CHARS[i])))
                }
            })
            .fold(Expr::Const(true), |a, b| {
                Expr::And(Box::new(a), Box::new(b))
            });
        expr = Expr::Or(Box::new(expr), Box::new(sub_expr));
    }
    expr.simplify_via_bdd()
}

fn expr_len(expr: &Expr<char>) -> usize {
    match expr {
        Expr::Terminal(_) => 0,
        Expr::Const(_) => 0,
        Expr::Not(expr) => expr_len(expr),
        Expr::And(expr1, expr2) => expr_len(expr1) + expr_len(expr2) + 1,
        Expr::Or(expr1, expr2) => expr_len(expr1) + expr_len(expr2) + 1,
    }
}

fn format_expr_postfix(expr: &Expr<char>) -> String {
    match expr {
        Expr::Terminal(c) => c.to_string(),
        Expr::Const(v) => v.to_string(),
        Expr::Not(expr) => format!("!{}", format_expr_postfix(expr)),
        Expr::And(expr1, expr2) => format!(
            "({} ^ {})",
            format_expr_postfix(expr1),
            format_expr_postfix(expr2)
        ),
        Expr::Or(expr1, expr2) => format!(
            "({} v {})",
            format_expr_postfix(expr1),
            format_expr_postfix(expr2)
        ),
    }
}

// MSB to LSB
pub(crate) fn binary_to_dec(input: &[bool]) -> u64 {
    let mut result = 0u64;
    for (i, bit) in input.iter().rev().enumerate() {
        if *bit {
            result |= 1 << i;
        }
    }
    result
}

#[test]
fn binary_to_dec_correctness() {
    let input = &[true, true, false, false];
    assert_eq!(binary_to_dec(input,), 12);
}

fn program_to_set(inputs: &[Vec<bool>], output: &[bool]) -> Vec<u64> {
    assert_eq!(inputs.len(), output.len());

    let mut trues = Vec::new();
    for (input, output) in inputs.iter().zip(output.iter()) {
        if !*output {
            continue;
        }

        trues.push(binary_to_dec(input));
    }

    trues
}

pub(crate) fn paper_repro() {
    // (!a & !b & !c) v (!a & b & c) v (a & !b & c) v (a & b & !c)
    let truth_table = [
        // FFF: true
        ([false, false, false], true),
        // FTT: true
        ([false, true, true], true),
        // TFT: true
        ([true, false, true], true),
        // TTF: true
        ([true, true, false], true),
        // all others false
        ([true, true, true], false),
        ([false, false, true], false),
        ([false, true, false], false),
        ([true, false, false], false),
    ];

    let inputs = truth_table
        .iter()
        .map(|(input, _)| input.to_vec())
        .collect::<Vec<_>>();
    let outputs = truth_table
        .iter()
        .map(|(_, output)| *output)
        .collect::<Vec<_>>();
    let expr = build_expr(&inputs, &outputs).simplify_via_bdd();
    println!("Paper Repro: {}", format_expr_postfix(&expr));
    let len = expr_len(&expr);
    println!("Paper Repro Len: {}", len);

    // let mut bdd = BDD::new();
    // let thing = bdd.from_expr(&expr);
    // println!("\n\n{}", bdd.to_dot(thing));

    // (a & ((!b & c) v (b & !c))) v (!a & ((!b & !c) v (b & c)))
    let theirs_minimal: Expr<char> = Expr::Or(
        Box::new(Expr::And(
            Box::new(Expr::Terminal('a')),
            Box::new(Expr::Or(
                Box::new(Expr::And(
                    Box::new(Expr::Not(Box::new(Expr::Terminal('b')))),
                    Box::new(Expr::Terminal('c')),
                )),
                Box::new(Expr::And(
                    Box::new(Expr::Terminal('b')),
                    Box::new(Expr::Not(Box::new(Expr::Terminal('c')))),
                )),
            )),
        )),
        Box::new(Expr::And(
            Box::new(Expr::Not(Box::new(Expr::Terminal('a')))),
            Box::new(Expr::Or(
                Box::new(Expr::And(
                    Box::new(Expr::Not(Box::new(Expr::Terminal('b')))),
                    Box::new(Expr::Not(Box::new(Expr::Terminal('c')))),
                )),
                Box::new(Expr::And(
                    Box::new(Expr::Terminal('b')),
                    Box::new(Expr::Terminal('c')),
                )),
            )),
        )),
    );
    let theirs_len = expr_len(&theirs_minimal);
    println!("Theirs Minimal: {}", format_expr_postfix(&theirs_minimal));
    println!("Theirs Minimal Len: {}", theirs_len);

    let theirs_opt = theirs_minimal.simplify_via_bdd();
    println!("Theirs Opt: {}", format_expr_postfix(&theirs_opt));
    let theirs_len = expr_len(&theirs_opt);
    println!("Theirs Opt Len: {}", theirs_len);
}

pub(crate) fn compute_boolean_complexities(dimensions: usize) -> Vec<usize> {
    // generate all permutations of `dimensions` boolean variables
    let inputs = (0..dimensions)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    // println!("inputs: {:?}", inputs);

    let output_count = inputs.len();
    // all possible permutations of [bool; output_count]
    let outputs = (0..output_count)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    // println!("outputs: {:?}", outputs);

    let mut lens = Vec::new();
    for output in outputs {
        let simplified_expr = build_expr(&inputs, &output);
        let len = expr_len(&simplified_expr);
        println!("{:?}: {}", simplified_expr, len);
        lens.push(len);
    }

    lens
}
