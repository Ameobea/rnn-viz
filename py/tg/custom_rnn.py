from functools import reduce
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
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
    elif isinstance(id, dict):
        id_type = id["id"]
        if id_type == "normal":
            return lambda shape: Tensor(np.random.normal(id["mean"], id["std"], shape).astype(np.float32))
        elif id_type == "uniform":
            return lambda shape: Tensor(np.random.uniform(id["low"], id["high"], shape).astype(np.float32))
        else:
            raise ValueError(f"Unknown initializer: {id}")
    else:
        raise ValueError(f"Unknown initializer: {id}")


class CustomRNNCell:
    def __init__(
        self,
        input_shape,
        output_dim: int,
        state_size: int,
        output_activation_id: Union[str, Dict[str, Any]] = "tanh",
        recurrent_activation_id: Union[str, Dict[str, Any]] = "tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        initial_state_initializer="glorot_uniform",
        output_kernel_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        output_bias_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        recurrent_kernel_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        recurrent_bias_regularizer: Optional[Callable[[Tensor], Tensor]] = None,
        trainable_initial_state=False,
        fedback_final_state_size: Optional[int] = None,
        feeding_back_state=False,
    ):
        self.trainable_weights: List[Tensor] = []

        self.input_dim = input_shape[-1]
        self.output_dim = output_dim
        self.state_size = state_size
        self.output_activation_id = output_activation_id
        self.recurrent_activation_id = recurrent_activation_id
        self.output_activation = build_activation(output_activation_id)
        self.recurrent_activation = build_activation(recurrent_activation_id)

        self.kernel_initializer = build_initializer(kernel_initializer)
        self.recurrent_initializer = build_initializer(recurrent_initializer)
        self.bias_initializer = build_initializer(bias_initializer)
        self.initial_state_initializer = build_initializer(initial_state_initializer)

        self.output_kernel_regularizer = output_kernel_regularizer
        self.output_bias_regularizer = output_bias_regularizer
        self.recurrent_kernel_regularizer = recurrent_kernel_regularizer
        self.recurrent_bias_regularizer = recurrent_bias_regularizer

        self.trainable_initial_state = trainable_initial_state

        input_size = (
            input_shape[-1]
            # + (self.state_size if not feeding_back_state else 0)
            + self.state_size
            + (fedback_final_state_size if fedback_final_state_size is not None else 0)
        )

        self.output_kernel = self.add_weight(
            shape=(input_size, self.output_dim),
            initializer=self.kernel_initializer,
        )

        self.recurrent_kernel = (
            self.add_weight(
                shape=(input_size, self.state_size),
                initializer=self.recurrent_initializer,
            )
            if self.state_size > 0
            else None
        )

        if use_bias:
            self.output_bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
            )

            self.recurrent_bias = (
                self.add_weight(
                    shape=(self.state_size,),
                    initializer=self.bias_initializer,
                )
                if self.state_size > 0
                else None
            )
        else:
            self.output_bias = None
            self.recurrent_bias = None

        self.initial_state = (
            self.add_weight(
                shape=(self.state_size,),
                initializer=self.initial_state_initializer,
                trainable=self.trainable_initial_state,
            )
            if self.state_size > 0
            else None
        )
        if self.initial_state is not None and not self.trainable_initial_state:
            self.initial_state.requires_grad = False

    def add_weight(
        self,
        shape,
        initializer: Callable[[List[int]], Tensor],
        trainable=True,
    ) -> Tensor:
        t = initializer(shape)
        if trainable:
            self.trainable_weights.append(t)
        return t

    def __call__(self, inputs: Tensor, prev_state: Tensor, extra_state: Tensor):
        if prev_state is not None and len(prev_state.shape) == 1:
            raise "prev_state must be a batch of states"
            # prev_state = prev_state.unsqueeze(0).repeat([inputs.shape[0], 1])

        combined_inputs = (
            inputs.cat(*[t for t in [prev_state, extra_state] if t is not None], dim=-1)
            if prev_state is not None or extra_state is not None
            else inputs
        )
        # print(
        #     inputs.shape,
        #     prev_state.shape if prev_state is not None else None,
        #     extra_state.shape if extra_state is not None else None,
        #     combined_inputs.shape,
        #     self.output_kernel.shape,
        # )
        output = combined_inputs.linear(self.output_kernel, self.output_bias)
        output = self.output_activation(output)

        if self.state_size == 0:
            return output, None

        new_state = combined_inputs.linear(self.recurrent_kernel, self.recurrent_bias)
        new_state = self.recurrent_activation(new_state)

        return output, new_state

        # def __call__(self, inputs: Tensor, prev_state: Tensor, extra_state: Tensor):
        #     if prev_state is not None and len(prev_state.shape) == 1:
        #         raise "prev_state must be a batch of states"
        #         # prev_state = prev_state.unsqueeze(0).repeat([inputs.shape[0], 1])

        #     input_dim = inputs.shape[-1]
        #     prev_state_dim = prev_state.shape[-1] if prev_state is not None else 0

        #     output_kernel_inputs = (
        #         self.output_kernel[:input_dim] if input_dim < self.output_kernel.shape[0] else self.output_kernel
        #     )
        #     output = inputs.dot(output_kernel_inputs)

        #     if prev_state is not None:
        #         output_kernel_prev_state = self.output_kernel[input_dim : input_dim + prev_state_dim]
        #         output = output + prev_state.dot(output_kernel_prev_state)
        #     if extra_state is not None:
        #         output_kernel_extra_state = self.output_kernel[input_dim + prev_state_dim :]
        #         output = output + extra_state.dot(output_kernel_extra_state)

        #     output = output.add(self.output_bias) if self.output_bias is not None else output
        #     output = self.output_activation(output)

        #     if self.state_size == 0:
        #         return output, None

        #     recurrent_kernel_inputs = (
        #         self.recurrent_kernel[:input_dim] if input_dim < self.recurrent_kernel.shape[0] else self.recurrent_kernel
        #     )
        #     new_state = inputs.dot(recurrent_kernel_inputs)

        #     if prev_state is not None:
        #         recurrent_kernel_prev_state = self.recurrent_kernel[input_dim : input_dim + prev_state_dim]
        #         new_state = new_state + prev_state.dot(recurrent_kernel_prev_state)
        #     if extra_state is not None:
        #         recurrent_kernel_extra_state = self.recurrent_kernel[input_dim + prev_state_dim :]
        #         new_state = new_state + extra_state.dot(recurrent_kernel_extra_state)

        #     new_state = new_state.add(self.recurrent_bias) if self.recurrent_bias is not None else new_state
        #     new_state = self.recurrent_activation(new_state)

        #     return output, new_state

    def get_initial_state(self, batch_size, dtype=None) -> Tensor:
        if self.initial_state is None:
            return None

        # tile initial state to batch size
        return self.initial_state.unsqueeze(0).repeat([batch_size, 1])

    def get_regularization_cost(self):
        cost = Tensor(0.0)
        if self.output_kernel_regularizer is not None:
            cost = cost + self.output_kernel_regularizer(self.output_kernel).sum()
        if self.output_bias_regularizer is not None:
            cost = cost + self.output_bias_regularizer(self.output_bias).sum()
        if self.recurrent_kernel_regularizer is not None and self.recurrent_kernel is not None:
            cost = cost + self.recurrent_kernel_regularizer(self.recurrent_kernel).sum()
        if self.recurrent_bias_regularizer is not None and self.recurrent_bias is not None:
            cost = cost + self.recurrent_bias_regularizer(self.recurrent_bias).sum()
        return cost


class CustomRNN:
    def __init__(
        self,
        *cells: CustomRNNCell,
        return_sequences=True,
        return_state=False,
        feedback_final_cell_state=False,
    ):
        CustomRNN.validate_cells(cells)
        self.cells = cells
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.feedback_final_cell_state = feedback_final_cell_state

    def __call__(self, inputs: Tensor):
        batch_size, seq_len = (inputs.shape[0], inputs.shape[1])
        states = [cell.get_initial_state(batch_size) for cell in self.cells]
        outputs = []
        for seq_ix in range(seq_len):
            inputs_for_timestep = inputs[:, seq_ix, :]
            new_states = []
            for cell_ix, cell, state in zip(range(len(self.cells)), self.cells, states):
                output, new_state = cell(
                    inputs_for_timestep,
                    # state if not self.feedback_final_cell_state or cell_ix != len(self.cells) - 1 else None,
                    state,
                    states[-1] if self.feedback_final_cell_state and cell_ix == 0 else None,
                )
                inputs_for_timestep = output
                new_states.append(new_state)
            outputs.append(inputs_for_timestep)
            states = new_states

        if self.return_sequences:
            output = Tensor.stack(outputs, dim=1)
        else:
            output = outputs[-1]

        if self.return_state:
            return output, states
        else:
            return output

    def get_trainable_params(self):
        trainable_params = []
        for cell in self.cells:
            trainable_params.extend(cell.trainable_weights)
        return trainable_params

    def get_regularization_loss(self) -> Tensor:
        return reduce(lambda a, b: a + b, [cell.get_regularization_cost() for cell in self.cells], Tensor(0.0))

    def validate_cells(cells: List[CustomRNNCell]):
        if len(cells) == 0:
            raise ValueError("cells must not be empty")

        input_dim = 0
        for i, cell in enumerate(list(cells)):
            if i == 0:
                input_dim = cell.output_dim
                continue

            if cell.input_dim != input_dim:
                raise ValueError(f"cell {i} has input_dim {cell.input_dim} but expected {input_dim}")
            input_dim = cell.output_dim

    def print_weights(self, dense: Optional[Linear] = None):
        for i, cell in enumerate(self.cells):
            print(f"[{i}] output weights:", cell.output_kernel.numpy())
            if cell.output_bias:
                print(f"[{i}] output bias:", cell.output_bias.numpy())
            if cell.recurrent_kernel:
                print(f"[{i}] recurrent weights:", cell.recurrent_kernel.numpy())
            if cell.recurrent_bias:
                print(f"[{i}] recurrent bias:", cell.recurrent_bias.numpy())
            if cell.initial_state is not None:
                print(f"[{i}] initial state:", cell.initial_state.numpy())
        if dense is not None:
            print("dense weights:", dense.weight.numpy())
            print("dense bias:", dense.bias.numpy())

    def dump_weights(self, post_layers: List[Tuple[Linear, Union[str, Dict[str, Any]]]], path: str):
        """
        Dumps weights as JSON to path
        """

        def tensor_to_list(t: Tensor):
            if t is None:
                return None
            return t.numpy().tolist()

        data = {
            "input_dim": self.cells[0].input_dim,
            "output_dim": post_layers[-1][0].weight.shape[0] if len(post_layers) > 0 else self.cells[-1].output_dim,
            "cells": [
                {
                    "state_size": cell.state_size,
                    "output_dim": cell.output_dim,
                    "output_kernel": tensor_to_list(cell.output_kernel),
                    "output_bias": tensor_to_list(cell.output_bias),
                    "recurrent_kernel": tensor_to_list(cell.recurrent_kernel),
                    "recurrent_bias": tensor_to_list(cell.recurrent_bias),
                    "initial_state": tensor_to_list(cell.initial_state),
                    "recurrent_activation": cell.recurrent_activation_id,
                    "output_activation": cell.output_activation_id,
                }
                for cell in self.cells
            ],
            "post_layers": [
                {
                    "input_dim": layer.weight.shape[1],
                    "output_dim": layer.weight.shape[0],
                    "weights": tensor_to_list(layer.weight),
                    "bias": tensor_to_list(layer.bias),
                    "activation": activation_id,
                }
                for layer, activation_id in post_layers
            ],
        }
        with open(path, "wt") as f:
            json.dump(data, f, indent=2)

        print(f"Dumped weights to {path}")
