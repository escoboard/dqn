import tensorflow as tf


class Graph:
    def __init__(self, actions, load_dir=None):
        self.actions = actions
        self.load_dir = load_dir
        self.graph = None
        self.input_layer = None

    def _get_weights(self, shape, name='weights'):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        tf.summary.histogram(name, weights)
        return weights

    def _get_bias(self, shape, name='bias'):
        with tf.name_scope(name):
            bias = tf.Variable(tf.constant(0.1, shape=shape))
        tf.summary.histogram(name, bias)
        return bias

    def _convolution_layer(self, input_layer, weights, name='conv_layer'):
        with tf.name_scope(name):
            return tf.nn.conv2d(input_layer, weights, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pooling_layer(self, input_layer, name='max_pool_layer'):
        with tf.name_scope(name):
            return tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _create_new_graph(self):
        with tf.name_scope("Input"):
            input_layer = tf.placeholder(tf.float32, shape=[None, 25600],name='id')
            input = tf.reshape(input_layer, [-1, 80, 320, 1])
        '''with tf.name_scope("Output"):
            action_value = tf.placeholder(tf.float32, shape=[None, self.actions])'''

        with tf.name_scope("Layer1"):
            conv_1_weights = self._get_weights([5, 5, 1, 32])
            conv_1_bias = self._get_bias([32])
            conv_layer_1 = tf.nn.relu(self._convolution_layer(input, conv_1_weights) + conv_1_bias)
            conv_pool_1 = self._max_pooling_layer(conv_layer_1)

        with tf.name_scope("Layer2"):
            conv_2_weights = self._get_weights([5, 5, 32, 64])
            conv_2_bias = self._get_bias([64])
            conv_layer_2 = tf.nn.relu(self._convolution_layer(conv_pool_1, conv_2_weights) + conv_2_bias)
            conv_pool_2 = self._max_pooling_layer(conv_layer_2)

        with tf.name_scope("Layer3"):
            conv_3_weights = self._get_weights([5, 5, 64, 128])
            conv_3_bias = self._get_bias([128])
            conv_layer_3 = tf.nn.relu(self._convolution_layer(conv_pool_2, conv_3_weights) + conv_3_bias)
            conv_pool_3 = self._max_pooling_layer(conv_layer_3)

        with tf.name_scope("Layer4"):
            conv_4_weights = self._get_weights([5, 5, 128, 256])
            conv_4_bias = self._get_bias([256])
            conv_layer_4 = tf.nn.relu(self._convolution_layer(conv_pool_3, conv_4_weights) + conv_4_bias)
            conv_pool_4 = self._max_pooling_layer(conv_layer_4)

        with tf.name_scope("HiddenLayer"):
            hidden_1_weights = self._get_weights([5 * 20 * 256, 2048])
            hidden_1_bias = self._get_bias([2048])
            hidden_layer_input = tf.reshape(conv_pool_4, [-1, 5 * 20 * 256])
            hidden_1_layer_output = tf.nn.relu(tf.matmul(hidden_layer_input, hidden_1_weights) + hidden_1_bias)

        with tf.name_scope("ReadoutLayer"):
            readout_weights = self._get_weights([2048, self.actions])
            readout_bias = self._get_bias([self.actions])

            action = tf.matmul(hidden_1_layer_output, readout_weights) + readout_bias
        '''
        with tf.name_scope("Loss"):
            loss = action_value - action
            tf.summary.histogram("Loss", loss)
        final_graph = tf.train.AdamOptimizer(1e-4).minimize(loss)
        '''

        return action, input_layer

    def get_graph(self):
        self.graph, self.input_layer = self._create_new_graph()
        if self.load_dir:
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, self.load_dir)
            tf.logging.INFO('Successfully loaded the graph')
        return self.graph

    def run_graph(self, session, image):
        print(session.run(self.graph, feed_dict={self.input_layer: image}))


if __name__ == '__main__':
    Graph(action=10).get_graph()
