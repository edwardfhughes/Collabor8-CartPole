import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


# Create model
def lstm_network(x, weights, biases):
    lstm_cell = rnn_cell.BasicLSTMCell(biases['b1'].get_shape().as_list()[0])
    state = lstm_cell.zero_state(1, tf.float32)
    rnn_outputs, state = rnn.rnn(lstm_cell, x, initial_state=state)
    layer_2 = tf.add(tf.matmul(rnn_outputs[-1], weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
