import numpy as np
from numba import jit

# def one_batch_examples(batch_size: int, seq_len: int):
#     # random numbers, either -1 or 1
#     inputs = np.random.choice([-1, 1], size=(batch_size, seq_len, 1)).astype(np.float32)
#     # train to return the previous number, or -1 if it's the first number
#     outputs = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
#     outputs[:, 1:, :] = inputs[:, :-1, :]
#     outputs[:, 0, :] = -1

#     return inputs, outputs


@jit
def one_val(prob=0.5):
    return 1 if np.random.random() < prob else -1


@jit
def xor(a, b):
    return 1 if (a == -1 and b == 1) or (a == 1 and b == -1) else -1


@jit
def and_(a, b):
    return 1 if a == 1 and b == 1 else -1


@jit
def or_(a, b):
    return 1 if a == 1 or b == 1 else -1


@jit
def xnor(a, b):
    return 1 if a == b else -1


@jit
def nand(a, b):
    return -1 if a == 1 and b == 1 else 1


@jit
def nor(a, b):
    return -1 if a == 1 or b == 1 else 1


@jit
def one_seq_examples(seq_len: int):
    inputs = np.zeros((seq_len, 3), dtype=np.float32)
    outputs = np.zeros((seq_len, 1), dtype=np.float32)

    # class Mode:
    #     Xor = -1
    #     And = 1
    #     Or = -2
    #     Nor = 2
    #     Nand = -3
    #     Xnor = 3

    # all_modes = [
    #     Mode.Xor,
    #     Mode.And,
    #     Mode.Nor,
    #     Mode.Nand,
    # ]  # Mode.Xnor]
    # mode = Mode.Xor
    # mode_index = 0

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


@jit
def one_batch_examples(batch_size: int, seq_len: int):
    inputs = []
    outputs = []
    for _ in range(batch_size):
        x, y = one_seq_examples(seq_len)
        inputs.append(x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs)


# def one_val_vectorized(prob=0.5, size=None):
#     return np.where(np.random.random(size) < prob, 1, -1)


# def one_seq_examples_v2(seq_len: int):
#     class Mode:
#         Xor = -1
#         And = 1
#         Or = -2
#         Nor = 2
#         Nand = -3
#         Xnor = 3

#     all_modes = [
#         Mode.Xor,
#         Mode.And,
#         Mode.Nor,
#         Mode.Nand,
#     ]
#     n_modes = len(all_modes)

#     change_mode = one_val_vectorized(0.3, seq_len)

#     mode_indices = (np.where(change_mode == 1)[0] % n_modes).astype(int)
#     modes = np.full(seq_len, all_modes[0])
#     modes[np.where(change_mode == 1)] = np.array(all_modes)[mode_indices]
#     print(modes)
#     exit(1)

#     inputs = np.column_stack(
#         [
#             change_mode,
#             one_val_vectorized(size=seq_len),
#             one_val_vectorized(size=seq_len),
#         ]
#     )

#     output = np.empty(seq_len)
#     for mode in all_modes:
#         mask = modes == mode
#         a, b = inputs[mask, 1], inputs[mask, 2]
#         if mode == Mode.Xor:
#             output[mask] = np.where(a != b, 1, -1)
#         elif mode == Mode.And:
#             output[mask] = np.where(np.logical_and(a == 1, b == 1), 1, -1)
#         elif mode == Mode.Or:
#             output[mask] = np.where(np.logical_or(a == 1, b == 1), 1, -1)
#         elif mode == Mode.Nor:
#             output[mask] = np.where(np.logical_or(a == 1, b == 1), -1, 1)
#         elif mode == Mode.Nand:
#             output[mask] = np.where(np.logical_and(a == 1, b == 1), -1, 1)

#     return inputs.astype(np.float32), output.reshape(-1, 1).astype(np.float32)


# def one_batch_examples_v2(batch_size: int, seq_len: int):
#     inputs, outputs = zip(*[one_seq_examples_v2(seq_len) for _ in range(batch_size)])
#     return np.array(inputs), np.array(outputs)
