from custom_rnn import CustomRNNCell, CustomRNN
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.jit import TinyJit

input_dim = 1
output_dim = 1
batch_size = 1024
seq_len = 16


def build_examples(batch_size: int, seq_len: int):
    # random numbers, either -1 or 1
    inputs = np.random.choice([-1, 1], size=(batch_size, seq_len, input_dim)).astype(np.float32)
    # train to return the previous number, or -1 if it's the first number
    outputs = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
    outputs[:, 1:, :] = inputs[:, :-1, :]
    outputs[:, 0, :] = -1

    return inputs, outputs


cells = CustomRNN(
    CustomRNNCell(
        input_shape=(
            batch_size,
            seq_len,
            1,
        ),
        output_dim=4,
        state_size=2,
        output_activation="ameo",
        recurrent_activation="ameo",
        trainable_initial_weights=True,
        use_bias=True,
    )
)
dense = Linear(cells.cell.output_dim, 1, bias=True)


def forward(x: Tensor) -> Tensor:
    y = cells(x)
    y = dense(y)
    return y


def compute_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return (y_pred - y_true).abs().mean()


trainable_params = cells.get_trainable_params()  # + [dense.weight, dense.bias]
opt = Adam(trainable_params, 0.05)


@TinyJit
def train_one_batch(x: Tensor, y: Tensor) -> Tensor:
    y_pred = forward(x)
    loss = compute_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.realize()


for i in range(800):
    x, y = build_examples(batch_size, seq_len)
    x, y = Tensor(x), Tensor(y)
    loss = train_one_batch(x, y)
    print(f"loss: {loss.numpy()}")

print("output weights:", cells.cell.output_kernel.numpy())
print("output bias:", cells.cell.output_bias.numpy() if cells.cell.output_bias else None)
print("recurrent weights:", cells.cell.recurrent_kernel.numpy())
print("recurrent bias:", cells.cell.recurrent_bias.numpy() if cells.cell.recurrent_bias else None)
print("initial state:", cells.cell.initial_state.numpy())
print("dense weights:", dense.weight.numpy())
print("dense bias:", dense.bias.numpy())

# eval test sequence
x, y = build_examples(1, seq_len)
x, y = Tensor(x), Tensor(y)
y_pred = forward(x)
print("input:", x.numpy())
print("expected output:", y.numpy())
print("predicted output:", y_pred.numpy())
