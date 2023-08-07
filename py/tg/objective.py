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


# @jit(nopython=True)
# def one_seq_examples(seq_len: int):
#     inputs = np.zeros((seq_len, 3), dtype=np.float32)
#     outputs = np.zeros((seq_len, 1), dtype=np.float32)

#     XOR = -1
#     AND = 1
#     OR = -2
#     NOR = 2
#     NAND = -3
#     XNOR = 3

#     all_modes = [
#         XOR,
#         AND,
#         NOR,
#         NAND,
#     ]
#     mode = XOR
#     mode_index = 0

#     for i in range(seq_len):
#         change_mode = one_val(0.3)
#         # change_dir = one_val()
#         if change_mode == 1:
#             # mode_index += 1 if change_dir == 1 else -1
#             mode_index += 1
#             mode_index = len(all_modes) - 1 if mode_index < 0 else mode_index
#             mode_index = 0 if mode_index >= len(all_modes) else mode_index
#             mode = all_modes[mode_index]

#         input_val = np.array(
#             [
#                 change_mode,
#                 # change_dir,
#                 one_val(),
#                 one_val(),
#             ]
#         ).astype(np.float32)
#         output_val = None
#         if mode == XOR:
#             output_val = np.array([xor(input_val[1], input_val[2])]).astype(np.float32)
#         elif mode == AND:
#             output_val = np.array([and_(input_val[1], input_val[2])]).astype(np.float32)
#         elif mode == OR:
#             output_val = np.array([or_(input_val[1], input_val[2])]).astype(np.float32)
#         elif mode == NOR:
#             output_val = np.array([nor(input_val[1], input_val[2])]).astype(np.float32)
#         elif mode == NAND:
#             output_val = np.array([nand(input_val[1], input_val[2])]).astype(np.float32)
#         elif mode == XNOR:
#             output_val = np.array([xnor(input_val[1], input_val[2])]).astype(np.float32)
#         else:
#             raise ValueError("Invalid mode")
#         inputs[i] = input_val
#         outputs[i] = output_val

#     return inputs, outputs

# 1 input, 1 output.
#
# Trained to estimate sin(2 * pi * x) for x in [-1, 1]
# Learned quite well with few neurons:
# {"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[{"weights":[{"weight":-1.302258849143982,"index":0}],"bias":1.0418460369110107,"name":"layer_0_output_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":0.44918179512023926,"index":0}],"bias":0.09361828863620758,"name":"layer_0_output_2","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-1.3898459672927856,"index":0}],"bias":-0.312580406665802,"name":"layer_0_output_3","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.5509219169616699,"index":0}],"bias":0.20590710639953613,"name":"layer_0_output_4","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-1.1935828924179077,"index":0}],"bias":0.34680449962615967,"name":"layer_0_output_5","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[],"bias":0.21351823210716248,"name":"layer_0_output_6","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.3268051743507385,"index":0}],"bias":-0.11300478875637054,"name":"layer_0_output_7","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}}],"recurrentNeurons":[],"stateNeurons":[],"outputDim":8},{"outputNeurons":[{"weights":[{"weight":0.6328311562538147,"index":2},{"weight":0.47096630930900574,"index":6},{"weight":-0.6744097471237183,"index":7}],"bias":0.10299275070428848,"name":"layer_1_output_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-1.0202131271362305,"index":3},{"weight":-0.8832091689109802,"index":4},{"weight":0.49551820755004883,"index":6}],"bias":0.3467811644077301,"name":"layer_1_output_1","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":0.7203356027603149,"index":0},{"weight":0.8442057371139526,"index":2},{"weight":-1.034474492073059,"index":3},{"weight":0.5440526604652405,"index":4},{"weight":-0.2977524697780609,"index":7}],"bias":-0.09384726732969284,"name":"layer_1_output_2","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,null,{"weights":[{"weight":1.2735137939453125,"index":0},{"weight":1.339267373085022,"index":5},{"weight":0.3336324393749237,"index":6}],"bias":0.5395767092704773,"name":"layer_1_output_5","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,null],"recurrentNeurons":[],"stateNeurons":[],"outputDim":8}],"postLayers":[{"neurons":[{"weights":[{"weight":-0.30023521184921265,"index":0},{"weight":0.6543983817100525,"index":1},{"weight":-0.6389914751052856,"index":2},{"weight":0.8249670267105103,"index":5}],"bias":-0.17620137333869934,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}
# @jit(nopython=True)
# def one_seq_examples(seq_len: int):
#     inputs = np.random.uniform(-1, 1, (seq_len, 1)).astype(np.float32)
#     outputs = np.sin(2 * np.pi * inputs).astype(np.float32)

#     return inputs, outputs

# asm interpreter
# @jit(nopython=True)
# def one_seq_examples(seq_len: int):
#     # inputs are instructions.
#     # instruction format:
#     # bit 0: opcode
#     #          -1: store
#     #          1: mov
#     # bit 1: rx register
#     # bit 2: tx register or immediate in the case of store
#     # output is the value written to the rx register

#     reg0 = -1
#     reg1 = -1

#     inputs = np.empty((seq_len, 3), dtype=np.float32)
#     outputs = np.empty((seq_len, 1), dtype=np.float32)

#     tx_regs = np.random.choice(np.array([-1, 1]), seq_len)
#     rx_regs = np.random.choice(np.array([-1, 1]), seq_len)
#     opcodes = np.random.choice(np.array([-1, 1]), seq_len)
#     immediates = np.random.choice(np.array([-1, 1]), seq_len)

#     for i in range(seq_len):
#         opcode = opcodes[i]
#         rx = rx_regs[i]
#         tx = tx_regs[i]

#         if opcode == -1:
#             # store
#             imm = immediates[i]
#             if rx == -1:
#                 reg0 = imm
#             else:
#                 reg1 = imm
#             inputs[i] = np.array([opcode, rx, imm])
#             outputs[i][0] = imm
#         else:
#             # mov
#             val = reg0 if rx == -1 else reg1
#             if tx == -1:
#                 reg0 = val
#             else:
#                 reg1 = val
#             inputs[i] = np.array([opcode, rx, tx])
#             outputs[i][0] = val

#     return inputs, outputs


# @jit(nopython=True)
# def one_seq_examples(seq_len: int, max_depth: int = 8):
#     # Inputs are sequences of parentheses.
#     # '(' is represented as -1, ')' is represented as 1.
#     # Outputs are binary sequences indicating whether each prefix of the
#     # input sequence is properly parenthesized.

#     inputs = np.empty((seq_len, 1), dtype=np.float32)
#     outputs = np.empty((seq_len, 1), dtype=np.float32)

#     depth = 0  # Current parentheses depth

#     for i in range(seq_len):
#         # Choose next character. Generate '(' if depth < max_depth
#         # and either depth is 0 (so we can't close yet) or with probability 0.5.
#         # Otherwise, generate ')'.
#         if depth < max_depth and (depth == 0 or np.random.choice(np.array([0, 1])) == 0):
#             inputs[i] = -1  # '('
#             depth += 1
#         else:
#             inputs[i] = 1  # ')'
#             depth -= 1

#         # Check if the prefix up to this point is valid (properly parenthesized).
#         # It's valid if depth >= 0 (every ')' had a matching '(').
#         outputs[i] = 1 if depth == 0 else -1

#     return inputs, outputs


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


# y_n =
#   -1 if n = 0
#   x_{n-1} otherwise
# def one_seq_examples(seq_len):
#     inputs = []
#     outputs = []
#     for i in range(seq_len):
#         input_val = np.array([one_val()]).astype(np.float32)
#         output_val = np.array([-1]).astype(np.float32) if i == 0 else inputs[-1]
#         inputs.append(input_val)
#         outputs.append(output_val)
#     inputs = np.array(inputs)
#     outputs = np.array(outputs)
#     return inputs, outputs


# y_n =
#   -1 if n < 2
#   x_{n-2} otherwise
# def one_seq_examples(seq_len):
#     inputs = []
#     outputs = []
#     for i in range(seq_len):
#         input_val = np.array([one_val()]).astype(np.float32)
#         output_val = np.array([-1]).astype(np.float32) if i == 0 or i == 1 else inputs[-2]
#         inputs.append(input_val)
#         outputs.append(output_val)
#     inputs = np.array(inputs)
#     outputs = np.array(outputs)
#     return inputs, outputs


# replace '1'->'111'
# Training input  : -1, -1, 1, -1, -1, -1, 1
# Expected outout : -1, -1, 1,  1,  1, -1, 1
def one_seq_examples(seq_len):
    inputs = []
    outputs = []
    replace_count = 0

    for i in range(seq_len):
        input_val = np.array([one_val()]).astype(np.float32)
        output_val = input_val

        if input_val == 1:
            replace_count = 3

        if replace_count > 0:
            output_val = np.array([1]).astype(np.float32)
            replace_count -= 1

        inputs.append(input_val)
        outputs.append(output_val)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


@jit(forceobj=True)
def one_batch_examples(batch_size: int, seq_len: int):
    inputs = []
    outputs = []
    for _ in range(batch_size):
        x, y = one_seq_examples(seq_len)
        inputs.append(x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs)
