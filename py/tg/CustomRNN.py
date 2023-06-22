from typing import Callable, Dict, List
from tinygrad.tensor import Tensor

def build_activation(id) -> Callable[[Tensor], Tensor]:
    if id=='tanh':
        return lambda x: x.tanh()
    elif id=='sigmoid':
        return lambda x: x.sigmoid()
    elif id=='relu':
        return lambda x: x.relu()
    elif id == 'linear' or id is None:
        return lambda x: x
    else:
        raise ValueError(f'Unknown activation: {id}')

def build_initializer(id) -> Callable[[List[int]], Tensor]:
    if id=='zeros':
        return lambda shape: Tensor.zeros(*shape)
    elif id=='ones':
        return lambda shape: Tensor.ones(*shape)
    elif id=='glorot_uniform':
        return lambda shape: Tensor.glorot_uniform(*shape)
    # elif id=='orthogonal':
    #     return lambda shape: Tensor.orthogonal(*shape)
    else:
        raise ValueError(f'Unknown initializer: {id}')

class CustomRNNCell:
    weights: Dict[str, Tensor] = {}
    trainable_weights = []
    regularizers = []

    def __init__(
        self,
        input_shape,
        output_dim,
        state_size,
        output_activation='tanh',
        recurrent_activation='tanh',
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        initial_state_initializer="glorot_uniform",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        trainable_initial_weights=False,
        **kwargs
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

        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)

        self.trainable_initial_weights = trainable_initial_weights

        self.output_kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.output_dim),
            initializer=self.kernel_initializer,
            name="kernel",
            # regularizer=self.kernel_regularizer,
        )

        self.recurrent_kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.state_size),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            # regularizer=self.recurrent_regularizer,
        )

        if self.use_bias:
            self.output_bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name="bias",
                # regularizer=self.bias_regularizer,
            )

            self.recurrent_bias = self.add_weight(
                shape=(self.state_size,),
                initializer=self.bias_initializer,
                name="recurrent_bias",
                # regularizer=self.bias_regularizer,
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

    def add_weight(self, shape, initializer: Callable[[List[int]], Tensor], name, trainable=True, regularizer=None) -> Tensor:
        t = initializer(shape)
        self.weights[name] = t
        if trainable:
            self.trainable_weights.append(t)
        # TODO: Handle regularizer
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
        return self.initial_state.unsqueeze(0).repeat([batch_size,1])


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
