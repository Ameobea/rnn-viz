import numpy as np


# def one_batch_examples(batch_size: int, seq_len: int):
#     # random numbers, either -1 or 1
#     inputs = np.random.choice([-1, 1], size=(batch_size, seq_len, 1)).astype(np.float32)
#     # train to return the previous number, or -1 if it's the first number
#     outputs = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
#     outputs[:, 1:, :] = inputs[:, :-1, :]
#     outputs[:, 0, :] = -1

#     return inputs, outputs


def one_val(prob=0.5):
    return 1 if np.random.random() < prob else -1


def xor(a, b):
    return 1 if (a == -1 and b == 1) or (a == 1 and b == -1) else -1


def and_(a, b):
    return 1 if a == 1 and b == 1 else -1


def or_(a, b):
    return 1 if a == 1 or b == 1 else -1


def xnor(a, b):
    return 1 if a == b else -1


def nand(a, b):
    return -1 if a == 1 and b == 1 else 1


def nor(a, b):
    return -1 if a == 1 or b == 1 else 1


def one_seq_examples(seq_len: int):
    inputs = []
    outputs = []

    class Mode:
        Xor = -1
        And = 1
        Or = -2
        Nor = 2
        Nand = -3
        Xnor = 3

    all_modes = [Mode.Xor, Mode.And, Mode.Nor, Mode.Nand, Mode.Xnor]
    mode = Mode.Xor
    mode_index = 0

    for i in range(seq_len):
        change_mode = one_val(0.8)
        change_dir = one_val()
        if change_mode == 1:
            mode_index += 1 if change_dir == 1 else -1
            mode_index = len(all_modes) - 1 if mode_index < 0 else mode_index
            mode_index = 0 if mode_index >= len(all_modes) else mode_index
            mode = all_modes[mode_index]

        input_val = np.array([change_mode, change_dir, one_val(), one_val()]).astype(np.float32)
        output_val = None
        if mode == Mode.Xor:
            output_val = np.array([xor(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == Mode.And:
            output_val = np.array([and_(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == Mode.Or:
            output_val = np.array([or_(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == Mode.Nor:
            output_val = np.array([nor(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == Mode.Nand:
            output_val = np.array([nand(input_val[1], input_val[2])]).astype(np.float32)
        elif mode == Mode.Xnor:
            output_val = np.array([xnor(input_val[1], input_val[2])]).astype(np.float32)
        else:
            raise ValueError("Invalid mode")
        inputs.append(input_val)
        outputs.append(output_val)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


def one_batch_examples(batch_size: int, seq_len: int):
    inputs = []
    outputs = []
    for i in range(batch_size):
        x, y = one_seq_examples(seq_len)
        inputs.append(x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs)
