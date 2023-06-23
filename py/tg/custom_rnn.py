from typing import Any, Callable, Dict, List, Optional, Union
from tinygrad.tensor import Tensor
import numpy as np
from glorot_normal import glorot_normal

from ameo_activation import mk_leaky_ameo, mk_interpolated_ameo


def build_activation(id: Union[str, Dict[str, Any]]) -> Callable[[Tensor], Tensor]:
    if isinstance(id, dict):
        if id["id"] == "leaky_ameo":
            leakyness = id["leakyness"]
            LeakyAmeo = mk_leaky_ameo(leakyness=leakyness)
            return lambda x: LeakyAmeo.apply(x)
        elif id["id"] == "interpolated_ameo":
            factor = id["factor"]
            leakyness = id["leakyness"]
            InterpolatedAmeo = mk_interpolated_ameo(factor, leakyness=leakyness)
            return lambda x: InterpolatedAmeo.apply(x)
        else:
            raise ValueError(f"Unknown activation: {id}")

    if id == "tanh":
        return lambda x: x.tanh()
    elif id == "sigmoid":
        return lambda x: x.sigmoid()
    elif id == "relu":
        return lambda x: x.relu()
    elif id == "linear" or id is None:
        return lambda x: x
    elif id == "ameo":
        LeakyAmeo = mk_leaky_ameo(leakyness=0.0)
        return lambda x: LeakyAmeo.apply(x)
    else:
        raise ValueError(f"Unknown activation: {id}")


def build_initializer(id) -> Callable[[List[int]], Tensor]:
    if id == "zeros":
        return lambda shape: Tensor.zeros(*shape)
    elif id == "ones":
        return lambda shape: Tensor.ones(*shape)
    elif id == "glorot_uniform":
        return lambda shape: Tensor.glorot_uniform(*shape)
    elif id == "glorot_normal":
        return lambda shape: Tensor(glorot_normal(shape))
    else:
        raise ValueError(f"Unknown initializer: {id}")


class CustomRNNCell:
    weights: Dict[str, Tensor] = {}
    trainable_weights = []

    def __init__(
        self,
        input_shape,
        output_dim,
        state_size,
        output_activation: Union[str, Dict[str, Any]] = "tanh",
        recurrent_activation: Union[str, Dict[str, Any]] = "tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        initial_state_initializer="glorot_uniform",
        output_kernel_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        output_bias_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        recurrent_kernel_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        recurrent_bias_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        trainable_initial_weights=False,
        **kwargs,
    ):
        self.output_dim = output_dim
        self.state_size = state_size
        self.output_activation = build_activation(output_activation)
        self.recurrent_activation = build_activation(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = build_initializer(kernel_initializer)
        self.recurrent_initializer = build_initializer(recurrent_initializer)
        self.bias_initializer = build_initializer(bias_initializer)
        self.initial_state_initializer = build_initializer(initial_state_initializer)

        self.output_kernel_regularizer = output_kernel_regularizer
        self.output_bias_regularizer = output_bias_regularizer
        self.recurrent_kernel_regularizer = recurrent_kernel_regularizer
        self.recurrent_bias_regularizer = recurrent_bias_regularizer

        self.trainable_initial_weights = trainable_initial_weights

        self.output_kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.output_dim),
            initializer=self.kernel_initializer,
            name="kernel",
        )

        self.recurrent_kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.state_size),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )

        if self.use_bias:
            self.output_bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name="bias",
            )

            self.recurrent_bias = self.add_weight(
                shape=(self.state_size,),
                initializer=self.bias_initializer,
                name="recurrent_bias",
            )
        else:
            self.output_bias = None
            self.recurrent_bias = None

        self.initial_state = self.add_weight(
            shape=(self.state_size,),
            initializer=self.initial_state_initializer,
            name="initial_state",
            trainable=self.trainable_initial_weights,
        )
        if not self.trainable_initial_weights:
            self.initial_state.requires_grad = False

    def add_weight(
        self,
        shape,
        initializer: Callable[[List[int]], Tensor],
        name,
        trainable=True,
    ) -> Tensor:
        t = initializer(shape)
        self.weights[name] = t
        if trainable:
            self.trainable_weights.append(t)
        return t

    def __call__(self, inputs: Tensor, prev_state: Tensor):
        if len(prev_state.shape) == 1:
            prev_state = prev_state.unsqueeze(0).repeat([inputs.shape[0], 1])

        combined_inputs = inputs.cat(prev_state, dim=-1)
        output = combined_inputs.linear(self.output_kernel, self.output_bias)
        output = self.output_activation(output)

        new_state = combined_inputs.linear(self.recurrent_kernel, self.recurrent_bias)
        new_state = self.recurrent_activation(new_state)

        return output, new_state

    def get_initial_state(self, batch_size, dtype=None) -> Tensor:
        # tile initial state to batch size
        return self.initial_state.unsqueeze(0).repeat([batch_size, 1])

    def get_regularization_cost(self):
        cost = Tensor(0.0)
        if self.output_kernel_regularizer is not None:
            cost = cost + self.output_kernel_regularizer(self.output_kernel).sum()
        if self.output_bias_regularizer is not None:
            cost = cost + self.output_bias_regularizer(self.output_bias).sum()
        if self.recurrent_kernel_regularizer is not None:
            cost = cost + self.recurrent_kernel_regularizer(self.recurrent_kernel).sum()
        if self.recurrent_bias_regularizer is not None:
            cost = cost + self.recurrent_bias_regularizer(self.recurrent_bias).sum()
        return cost


class CustomRNN:
    # TODO: Support multiple cells
    def __init__(self, cell: CustomRNNCell, return_sequences=True, return_state=False, **kwargs):
        super(CustomRNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.states = None

    def __call__(self, inputs: Tensor):
        batch_size, seq_len = (inputs.shape[0], inputs.shape[1])
        self.states = self.cell.get_initial_state(batch_size)
        outputs = []
        for seq_ix in range(seq_len):
            inputs_for_timestep = inputs[:, seq_ix, :]
            output, self.states = self.cell(inputs_for_timestep, self.states)
            outputs.append(output)

        if self.return_sequences:
            output = Tensor.stack(outputs, dim=1)
        else:
            output = outputs[-1]

        if self.return_state:
            return output, self.states
        else:
            return output

    def get_trainable_params(self):
        return self.cell.trainable_weights

    def get_regularization_loss(self):
        return self.cell.get_regularization_cost()
