from typing import Optional
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers


class CustomRNNCell(Layer):
    def __init__(
        self,
        output_dim: int,
        state_size: int,
        activation=tf.nn.tanh,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        initial_state_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        trainable_initial_weights=False,
        fedback_final_state_size: Optional[int] = None,
        **kwargs
    ):
        super(CustomRNNCell, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.state_size = state_size
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.initial_state_initializer = initializers.get(initial_state_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.trainable_initial_weights = trainable_initial_weights
        self.feeding_back_state = False
        self.fedback_final_state_size = fedback_final_state_size

    def build(self, input_shape):
        input_size = (
            input_shape[-1]
            + (self.state_size if not self.feeding_back_state else 0)
            + (self.fedback_final_state_size if self.fedback_final_state_size is not None else 0)
        )

        self.kernel = self.add_weight(
            shape=(input_size, self.output_dim),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
        )

        self.recurrent_kernel = self.add_weight(
            shape=(input_size, self.state_size),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            regularizer=self.recurrent_regularizer,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
            )

            self.recurrent_bias = self.add_weight(
                shape=(self.state_size,),
                initializer=self.bias_initializer,
                name="recurrent_bias",
                regularizer=self.bias_regularizer,
            )

        self.initial_state = self.add_weight(
            shape=(self.state_size,),
            initializer=self.initial_state_initializer,
            name="initial_state",
            trainable=self.trainable_initial_weights,
        )

    def call(self, inputs, prev_output, extra_state):
        if prev_output is not None and len(prev_output.shape) == 1:
            print("needs expanding")
            prev_output = tf.repeat(tf.expand_dims(prev_output, 0), repeats=tf.shape(inputs)[0], axis=0)

        combined_inputs = tf.concat([t for t in [inputs, prev_output, extra_state] if t is not None], axis=-1)
        print(
            inputs.shape,
            prev_output.shape if prev_output is not None else None,
            extra_state.shape if extra_state is not None else None,
            combined_inputs.shape,
            self.kernel.shape,
        )
        h = tf.matmul(combined_inputs, self.kernel)
        if self.use_bias:
            h = tf.nn.bias_add(h, self.bias)

        output = self.activation(h)

        h_recurrent = tf.matmul(combined_inputs, self.recurrent_kernel)
        if self.use_bias:
            h_recurrent = tf.nn.bias_add(h_recurrent, self.recurrent_bias)

        new_state = self.activation(h_recurrent)

        return output, new_state

    def get_initial_state(self, batch_size, dtype=None):
        # tile initial state to batch size
        initial_state = tf.tile(tf.expand_dims(self.initial_state, 0), [batch_size, 1])
        return initial_state

    def get_config(self):
        config = super(CustomRNNCell, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "state_size": self.state_size,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "trainable_initial_weights": self.trainable_initial_weights,
            }
        )
        return config


class CustomRNN(Layer):
    def __init__(
        self,
        *cells: CustomRNNCell,
        return_sequences=False,
        return_state=False,
        feedback_final_cell_state=False,
        **kwargs
    ):
        super(CustomRNN, self).__init__(**kwargs)
        self.cells = cells
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.feedback_final_cell_state = feedback_final_cell_state

    def build(self, input_shape):
        if len(input_shape) != 3:
            print("input shape: ", input_shape)
            raise ValueError("Input shape should be (batch_size, sequence_length, input_dim)")
        for cell in self.cells:
            cell.build(input_shape)
            input_shape = (input_shape[0], input_shape[1], cell.output_dim)
        self.built = True

    def call(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], inputs.shape[1]
        states = [cell.get_initial_state(batch_size) for cell in self.cells]
        outputs = []
        for seq_ix in range(seq_len):
            inputs_for_timestep = inputs[:, seq_ix, :]
            new_states = []
            # for cell_ix, cell, state in zip(range(len(self.cells)), self.cells, states):
            for cell_ix in range(len(self.cells)):
                cell = self.cells[cell_ix]
                state = states[cell_ix]
                output, new_state = cell.call(
                    inputs_for_timestep,
                    state,
                    states[-1] if self.feedback_final_cell_state and cell_ix == 0 else None,
                )
                inputs_for_timestep = output
                new_states.append(new_state)
            outputs.append(inputs_for_timestep)
            states = new_states

        if self.return_sequences:
            output = tf.stack(outputs, axis=1)
        else:
            output = outputs[-1]

        if self.return_state:
            return output, states
        else:
            return output
