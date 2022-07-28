use boolean_expression::Expr;
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
                    Expr::Not(box Expr::Terminal(CHARS[i]))
                }
            })
            .fold(Expr::Const(true), |a, b| Expr::And(box a, box b));
        expr = Expr::Or(box expr, box sub_expr);
    }
    expr.simplify_via_bdd()
}

fn expr_len(expr: &Expr<char>) -> usize {
    match expr {
        Expr::Terminal(_) => 1,
        Expr::Const(_) => 0,
        Expr::Not(expr) => expr_len(expr),
        Expr::And(expr1, expr2) => expr_len(expr1) + expr_len(expr2),
        Expr::Or(expr1, expr2) => expr_len(expr1) + expr_len(expr2),
    }
}

pub(crate) fn compute_boolean_complexities(dimensions: usize) -> Vec<usize> {
    // generate all permutations of `dimensions` boolean variables
    let inputs = (0..dimensions)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    println!("inputs: {:?}", inputs);

    let output_count = inputs.len();
    // all possible permutations of [bool; output_count]
    let outputs = (0..output_count)
        .map(|_| [false, true].into_iter())
        .multi_cartesian_product()
        .collect_vec();
    println!("outputs: {:?}", outputs);

    let mut lens = Vec::new();
    for output in outputs {
        let simplified_expr = build_expr(&inputs, &output);
        let len = expr_len(&simplified_expr);
        println!("{:?}: {}", simplified_expr, len);
        lens.push(len);
    }

    lens
}
