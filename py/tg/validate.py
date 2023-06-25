from typing import Callable, Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
import numpy as np


def f32_to_int(x: Tensor) -> Tensor:
    # return (x + 0.5).floor().cast(dtypes.int32)
    return (x > 0.0).cast(dtypes.float32).mul(2.0).sub(1.0).cast(dtypes.int32)


def validate(
    one_batch_examples: Callable[
        [int, int],
        Tuple[np.ndarray, np.ndarray],
    ],
    forward: Callable[[Tensor], Tensor],
    seq_len: int,
    test_count: int = 50000,
):
    batch_size = int(test_count / 50)
    batch_count = int(test_count / batch_size)

    print("\n\n\nRunning Validation...\n\n")

    for batch_ix in range(batch_count):
        x, expected = one_batch_examples(batch_size, seq_len)

        y_pred = f32_to_int(forward(Tensor(x, dtype=dtypes.float32)))
        expected = f32_to_int(Tensor(expected, dtype=dtypes.float32))

        incorrect_count = (y_pred.realize() != expected.realize()).sum().numpy()

        if incorrect_count == 0:
            if batch_ix % 10 == 0:
                print(f"validation PASS; batch {batch_ix} of {batch_count}")
            continue

        total_count = batch_size * seq_len
        acc = 1 - (incorrect_count / total_count)
        print("incorrect_count", incorrect_count)
        print(f"validation FAIL; accuracy: {acc*100:.2f}%")

        # find index of sequence with first error
        for seq_ix in range(batch_size):
            seq_expected = expected[seq_ix]
            seq_pred = y_pred[seq_ix]

            if (seq_expected != seq_pred).sum().numpy() > 0:
                print(f"expected: {seq_expected.numpy()}")
                print(f"actual: {seq_pred.numpy()}")

                seq_expected = seq_expected.numpy()
                seq_pred = seq_pred.numpy()

                for i in range(seq_len):
                    if seq_expected[i] != seq_pred[i]:
                        print(
                            f"first error at index {i}; expected {seq_expected[i]}, got {seq_pred[i]}"
                        )
                        break

                return

    print(f"VALIDATION PASS for {test_count} examples")
