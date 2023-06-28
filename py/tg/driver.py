from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import queue
from typing import Tuple

from custom_rnn import CustomRNNCell, CustomRNN
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.jit import TinyJit
from sparse_regularizer import SparseRegularizer
from objective import one_batch_examples
from validate import validate


def data_gen_worker(
    data_queue: queue.Queue[Tuple[np.ndarray, np.ndarray]],
    done: queue.Queue[bool],
    batch_size: int,
    seq_len: int,
):
    while True:
        x, y = one_batch_examples(batch_size, seq_len)
        while True:
            try:
                data_queue.put((x, y), block=True, timeout=0.1)
                break
            except queue.Full:
                if not done.empty():
                    return


if __name__ == "__main__":
    learning_rate = 0.004
    seq_len = 40
    input_dim = one_batch_examples(1, seq_len)[0].shape[-1]
    output_dim = one_batch_examples(1, seq_len)[1].shape[-1]
    batch_size = 1024 * 4

    np.set_printoptions(suppress=True)

    init = "glorot_normal"
    # init = {"id": "uniform", "low": -1, "high": 1}

    reg = SparseRegularizer(intensity=0.1, threshold=0.025, steepness=25, l1=0.001)
    activation = {"id": "interpolated_ameo", "factor": 0.5, "leakyness": 0.01}

    rnn = CustomRNN(
        CustomRNNCell(
            input_shape=(
                batch_size,
                seq_len,
                input_dim,
            ),
            output_dim=16,
            state_size=10,
            output_activation_id=activation,
            recurrent_activation_id=activation,
            trainable_initial_weights=True,
            use_bias=True,
            output_kernel_regularizer=reg,
            recurrent_kernel_regularizer=reg,
            kernel_initializer=init,
            bias_initializer=init,
            initial_state_initializer=init,
            # output_bias_regularizer=reg,
            # recurrent_bias_regularizer=reg,
            cell_ix=0,
        ),
        CustomRNNCell(
            input_shape=(
                batch_size,
                seq_len,
                16,
            ),
            output_dim=16,
            state_size=10,
            output_activation_id=activation,
            recurrent_activation_id=activation,
            trainable_initial_weights=True,
            use_bias=True,
            output_kernel_regularizer=reg,
            recurrent_kernel_regularizer=reg,
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_normal",
            initial_state_initializer="glorot_normal",
            # output_bias_regularizer=reg,
            # recurrent_bias_regularizer=reg,
            cell_ix=1,
        ),
    )

    dense = Linear(rnn.cells[-1].output_dim, 1, bias=True) if rnn.cells[-1].output_dim != output_dim else None

    def forward(x: Tensor) -> Tensor:
        y = rnn(x)
        if y.shape[-1] != output_dim:
            y = dense(y)  # .tanh()
        return y

    def compute_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return (y_pred - y_true).pow(2).mean()

    trainable_params = rnn.get_trainable_params() + ([dense.weight, dense.bias] if dense else [])
    opt = Adam(
        trainable_params,
        learning_rate,
    )

    def mk_train_one_batch():
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

        return train_one_batch

    multiprocessing.freeze_support()

    data_queue = multiprocessing.Manager().Queue(maxsize=10)
    done = multiprocessing.Manager().Queue(maxsize=10)
    data_gen_worker_count = 12

    # Start data generation in worker threads
    with ProcessPoolExecutor(max_workers=data_gen_worker_count) as executor:
        for _ in range(data_gen_worker_count):
            executor.submit(data_gen_worker, data_queue, done, batch_size, seq_len)

        # Training loop
        train_one_batch = mk_train_one_batch()
        for i in range(5000):
            if i == 500:
                reg.intensity *= 0.8
                opt.lr *= 0.8
                train_one_batch = mk_train_one_batch()
            if i == 1000:
                reg.intensity *= 0.6
                opt.lr *= 0.6
                train_one_batch = mk_train_one_batch()
            if i == 2500:
                # reg.intensity *= 0.5
                opt.lr *= 0.5
                train_one_batch = mk_train_one_batch()

            x, y = data_queue.get()
            x, y = Tensor(x), Tensor(y)
            loss = train_one_batch(x, y)
            print(f"[{i}]: loss: {loss.numpy()}")

        done.put(True)

    print("Done training")
    rnn.print_weights(dense)
    homedir = os.path.expanduser("~")
    rnn.dump_weights([(dense, "linear")] if dense else [], f"{homedir}/Downloads/weights.json")

    validate(one_batch_examples, forward, 40)
