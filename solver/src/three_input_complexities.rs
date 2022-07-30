/// Numbering all 3-input boolean functions by treating their output truth tables as binary numbers.  Then,
/// compute the simplest boolean expression that represents that function using only AND, OR, and NOT with
/// the minimal number of ANDs and ORs.  NOTs are treated as free.  Then, take the count of ANDs and ORs in
/// that expression.
///
/// Generated using code from here:
/// http://oeis.org/A056287/a056287.txt
pub const THREE_INPUT_BOOLEAN_COMPLEXITIES: [u8; 256] = [
    1, 2, 2, 1, 2, 1, 4, 2, 2, 4, 1, 2, 1, 2, 2, 0, 2, 1, 4, 2, 4, 2, 7, 4, 5, 4, 4, 3, 4, 3, 5, 2,
    2, 4, 1, 2, 5, 4, 4, 3, 4, 7, 2, 4, 4, 5, 3, 2, 1, 2, 2, 0, 4, 3, 5, 2, 4, 5, 3, 2, 3, 4, 4, 1,
    2, 4, 5, 4, 1, 2, 4, 3, 4, 7, 4, 5, 2, 4, 3, 2, 1, 2, 4, 3, 2, 0, 5, 2, 4, 5, 3, 4, 3, 2, 4, 1,
    4, 7, 4, 5, 4, 5, 3, 4, 7, 9, 5, 7, 5, 7, 4, 4, 2, 4, 3, 2, 3, 2, 4, 1, 5, 7, 4, 4, 4, 4, 5, 2,
    2, 5, 4, 4, 4, 4, 7, 5, 1, 4, 2, 3, 2, 3, 4, 2, 4, 4, 7, 5, 7, 5, 9, 7, 4, 3, 5, 4, 5, 4, 7, 4,
    1, 4, 2, 3, 4, 3, 5, 4, 2, 5, 0, 2, 3, 4, 2, 1, 2, 3, 4, 2, 5, 4, 7, 4, 3, 4, 2, 1, 4, 5, 4, 2,
    1, 4, 4, 3, 2, 3, 5, 4, 2, 5, 3, 4, 0, 2, 2, 1, 2, 3, 5, 4, 4, 2, 7, 4, 3, 4, 4, 5, 2, 1, 4, 2,
    2, 5, 3, 4, 3, 4, 4, 5, 4, 7, 2, 4, 2, 4, 1, 2, 0, 2, 2, 1, 2, 1, 4, 2, 2, 4, 1, 2, 1, 2, 2, 1,
];
