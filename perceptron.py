import tensorflow as tf


class MultilayerPerceptron:

    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev=0.01))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([n_output]))
        }
        self.reg = 0

    def evaluate(self, x):
        # Hidden layer with tanh activation
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden layer with tanh activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.tanh(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        reg = self.reg * (tf.reduce_sum(tf.square(self.weights['h1']))
                          + tf.reduce_sum(tf.square(self.weights['h2']))
                          + tf.reduce_sum(tf.square(self.weights['out'])))
        return out_layer, reg
