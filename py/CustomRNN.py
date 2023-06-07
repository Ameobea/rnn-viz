import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers


class CustomRNNCell(Layer):
    def __init__(
        self,
        output_dim,
        state_size,
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

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.output_dim),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
        )

        self.recurrent_kernel = self.add_weight(
            shape=(input_shape[-1] + self.state_size, self.state_size),
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

    def call(self, inputs, states):
        prev_output = states[0]
        if len(prev_output.shape) == 1:
            prev_output = tf.repeat(tf.expand_dims(prev_output, 0), repeats=tf.shape(inputs)[0], axis=0)

        combined_inputs = tf.concat([inputs, prev_output], axis=1)
        h = tf.matmul(combined_inputs, self.kernel)
        if self.use_bias:
            h = tf.nn.bias_add(h, self.bias)

        output = self.activation(h)

        h_recurrent = tf.matmul(combined_inputs, self.recurrent_kernel)
        if self.use_bias:
            h_recurrent = tf.nn.bias_add(h_recurrent, self.recurrent_bias)

        new_state = self.activation(h_recurrent)

        return output, [new_state]

    def get_initial_state(self, batch_size, dtype=None):
        # tile initial state to batch size
        initial_state = tf.tile(tf.expand_dims(self.initial_state, 0), [batch_size, 1])
        return [initial_state]

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
    def __init__(self, cell, return_sequences=False, return_state=False, **kwargs):
        super(CustomRNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.states = None

    def build(self, input_shape):
        self.cell.build(input_shape)
        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.states = self.cell.get_initial_state(batch_size)
        outputs = []
        for t in range(inputs.shape[1]):
            output, self.states = self.cell(inputs[:, t], self.states)
            outputs.append(output)

        if self.return_sequences:
            output = tf.stack(outputs, axis=1)
        else:
            output = outputs[-1]

        if self.return_state:
            return output, self.states
        else:
            return output
