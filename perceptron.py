import tensorflow as tf


class MultilayerPerceptron:

    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)), # could use xavier_optimiser (what's the advantage?)
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev=0.1))
        }
        """
        self.biases = {
            'b1': tf.Variable(tf.random_uniform([n_hidden_1], minval=-0.1, maxval=0.1)),
            'b2': tf.Variable(tf.random_uniform([n_hidden_2], minval=-0.1, maxval=0.1)),
            'out': tf.Variable(tf.random_uniform([n_output], minval=-0.1, maxval=0.1))
        }
        """

    def evaluate(self, x):
        # Hidden layer with sigmoid activation
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), 0) #self.biases['b1']
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden layer with sigmoid activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), 0) #self.biases['b2']
        layer_2 = tf.nn.tanh(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, self.weights['out'])#  + self.biases['out']
        return out_layer
