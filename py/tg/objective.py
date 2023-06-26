import numpy as np
from numba import jit


@jit(nopython=True)
def one_val(prob=0.5):
    return 1 if np.random.random() < prob else -1


@jit(nopython=True)
def xor(a, b):
    return 1 if (a == -1 and b == 1) or (a == 1 and b == -1) else -1


@jit(nopython=True)
def and_(a, b):
    return 1 if a == 1 and b == 1 else -1


@jit(nopython=True)
def or_(a, b):
    return 1 if a == 1 or b == 1 else -1


@jit(nopython=True)
def xnor(a, b):
    return 1 if a == b else -1


@jit(nopython=True)
def nand(a, b):
    return -1 if a == 1 and b == 1 else 1


@jit(nopython=True)
def nor(a, b):
    return -1 if a == 1 or b == 1 else 1


@jit(nopython=True)
def one_seq_examples(seq_len: int):
    inputs = np.zeros((seq_len, 3), dtype=np.float32)
    outputs = np.zeros((seq_len, 1), dtype=np.float32)

    XOR = -1
    AND = 1
    OR = -2
    NOR = 2
    NAND = -3
    XNOR = 3

    all_modes = [
        XOR,
        AND,
        NOR,
        NAND,
    ]
    mode = XOR
    mode_index = 0

    for i in range(seq_len):
        change_mode = one_val(0.3)
        # change_dir = one_val()
        if change_mode == 1:
            # mode_index += 1 if change_dir == 1 else -1
            mode_index += 1
            mode_index = len(all_modes) - 1 if mode_index < 0 else mode_index
            mode_index = 0 if mode_index >= len(all_modes) else mode_index
            mode = all_modes[mode_index]

        input_val = np.array(
            [
                change_mode,
                # change_dir,
                one_val(),
                one_val(),
            ]
        ).astype(np.float32)
        output_val = None
        if mode == XOR:
            output_val = np.array([xor(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == AND:
            output_val = np.array([and_(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == OR:
            output_val = np.array([or_(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == NOR:
            output_val = np.array([nor(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == NAND:
            output_val = np.array([nand(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == XNOR:
            output_val = np.array([xnor(input_val[1], input_val[2])]).astype(np.float32)
        else:
            raise ValueError("Invalid mode")
        inputs[i] = input_val
        outputs[i] = output_val

    return inputs, outputs


# # debug: passthru inputs
# def one_seq_examples(seq_len):
#     inputs = []
#     outputs = []
#     for i in range(seq_len):
#         input_val = np.array([one_val()]).astype(np.float32)
#         output_val = input_val
#         inputs.append(input_val)
#         outputs.append(output_val)
#     inputs = np.array(inputs)
#     outputs = np.array(outputs)
#     return inputs, outputs


@jit(forceobj=True)
def one_batch_examples(batch_size: int, seq_len: int):
    inputs = []
    outputs = []
    for _ in range(batch_size):
        x, y = one_seq_examples(seq_len)
        inputs.append(x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs)
