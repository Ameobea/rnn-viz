from custom_rnn import CustomRNNCell, CustomRNN
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.jit import TinyJit
from sparse_regularizer import SparseRegularizer
from objective import one_batch_examples
from validate import validate


learning_rate = 0.003
seq_len = 18
input_dim = one_batch_examples(1, seq_len)[0].shape[-1]
output_dim = one_batch_examples(1, seq_len)[1].shape[-1]
batch_size = 1024 * 4


reg = SparseRegularizer(intensity=0.4, threshold=0.025, steepness=25, l1=0.001)
activation = {"id": "interpolated_ameo", "factor": 0.6, "leakyness": 1}

rnn = CustomRNN(
    CustomRNNCell(
        input_shape=(
            batch_size,
            seq_len,
            input_dim,
        ),
        output_dim=64,
        state_size=64,
        output_activation=activation,
        recurrent_activation=activation,
        trainable_initial_weights=True,
        use_bias=True,
        output_kernel_regularizer=reg,
        recurrent_kernel_regularizer=reg,
        kernel_initializer="glorot_normal",
        bias_initializer="glorot_normal",
        initial_state_initializer="glorot_normal",
        # output_bias_regularizer=reg,
        # recurrent_bias_regularizer=reg,
    )
)
dense = Linear(rnn.cells[-1].output_dim, 1, bias=True)


def forward(x: Tensor) -> Tensor:
    y = rnn(x)
    y = dense(y)
    return y


def compute_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return (y_pred - y_true).pow(2).mean()


trainable_params = rnn.get_trainable_params() + [dense.weight, dense.bias]
opt = Adam(
    trainable_params,
    learning_rate,
)


@TinyJit
def train_one_batch(x: Tensor, y: Tensor) -> Tensor:
    y_pred = forward(x)
    raw_loss = compute_loss(y_pred, y)
    reg_loss = rnn.get_regularization_loss() + reg(dense.weight)
    loss = raw_loss + reg_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return raw_loss.reshape((1,)).cat(reg_loss.reshape((1,))).realize()


np.set_printoptions(suppress=True)

for i in range(500):
    x, y = one_batch_examples(batch_size, seq_len)
    x, y = Tensor(x), Tensor(y)
    loss = train_one_batch(x, y)
    print(f"loss: {loss.numpy()}")
for i, cell in enumerate(rnn.cells):
    print(f"[{i}] output weights:", cell.output_kernel.numpy())
    if cell.output_bias:
        print(f"[{i}] output bias:", cell.output_bias.numpy())
    print(f"[{i}] recurrent weights:", cell.recurrent_kernel.numpy())
    if cell.recurrent_bias:
        print(f"[{i}] recurrent bias:", cell.recurrent_bias.numpy())
    print(f"[{i}] initial state:", cell.initial_state.numpy())
print("dense weights:", dense.weight.numpy())
print("dense bias:", dense.bias.numpy())


validate(one_batch_examples, forward, 40)
