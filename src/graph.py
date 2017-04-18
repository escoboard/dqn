import tensorflow as tf

FLAGS = None

class Graph:
    def __init__(self, actions, tf_session, load_dir=None):
        self.actions = actions
        self.load_dir = load_dir
        self.tf_session = tf_session
        self.graph = None

    def _get_weights(self, shape, name='weights'):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.005))
        tf.summary.histogram(name, weights)
        return weights

    def _get_bias(self, shape, name='bias'):
        with tf.name_scope(name):
            bias = tf.Variable(tf.constant(0.005, shape=shape))
        tf.summary.histogram(name, bias)
        return bias

    def _convolution_layer(self, input_layer ,weights ,strides  = [1, 1, 1, 1] ,name='conv_layer'):
        with tf.name_scope(name):
            return tf.nn.conv2d(input_layer, weights, strides=strides, padding='VALID')

    def _max_pooling_layer(self, input_layer, name='max_pool_layer'):
        with tf.name_scope(name):
            return tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _create_new_graph(self):
        with tf.name_scope('Input'):
            input = tf.placeholder(tf.float32, shape=[None, 84,84,4], name='id')

        with tf.name_scope('Updated_value'):
            updated_action = tf.placeholder(tf.float32, shape=[None, self.actions])

        with tf.name_scope('Layer1'):
            conv_1_weights = self._get_weights([8, 8, 4, 32])
            conv_1_bias = self._get_bias([32])
            conv_layer_1 = tf.nn.relu(self._convolution_layer(input, conv_1_weights,strides = [1,4,4,1]) + conv_1_bias)


        with tf.name_scope('Layer2'):
            conv_2_weights = self._get_weights([4, 4, 32, 64])
            conv_2_bias = self._get_bias([64])
            conv_layer_2 = tf.nn.relu(self._convolution_layer(conv_layer_1, conv_2_weights,[1,2,2,1]) + conv_2_bias)


        with tf.name_scope('Layer3'):
            conv_3_weights = self._get_weights([3, 3, 64, 64])
            conv_3_bias = self._get_bias([64])
            conv_layer_3 = tf.nn.relu(self._convolution_layer(conv_layer_2, conv_3_weights,[1,1,1,1]) + conv_3_bias)


        with tf.name_scope('HiddenLayer'):
            hidden_1_weights = self._get_weights([7*7*64, 512])
            hidden_1_bias = self._get_bias([512])
            hidden_layer_input = tf.reshape(conv_layer_3, [-1, 7*7*64])
            hidden_1_layer_output = tf.nn.relu(tf.matmul(hidden_layer_input, hidden_1_weights) + hidden_1_bias)

        with tf.name_scope('ReadoutLayer'):
            readout_weights = self._get_weights([512, self.actions])
            readout_bias = self._get_bias([self.actions])

        with tf.name_scope('Output_Distribution'):
            action_value = tf.nn.softmax(tf.matmul(hidden_1_layer_output, readout_weights) + readout_bias)
            tf.summary.histogram('output', action_value)

        with tf.name_scope("Loss"):
            diff =  updated_action - action_value 
            tf.summary.histogram("Diff", diff)
            loss = tf.reduce_mean(diff)
            tf.summary.scalar("Loss", loss)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster))
            train_op = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

        return train_op, input, action_value, updated_action, loss

    def _load_graph(self, load_dir):
        saver = tf.train.Saver()

        saver.restore(self.tf_session, self.load_dir)
        tf.logging.INFO('Successfully loaded the graph')

    def save_graph(self, save_dir):
        saver = tf.train.Saver()
        saver.save(self.tf_session, save_dir)
        #tf.logging.INFO('Successfully saved the graph at %s' % save_dir)

    def get_graph(self):
        self.graph, input_layer, action_value, updated_action, loss = self._create_new_graph()
        if self.load_dir:
            self._load_graph(self.load_dir)
        return self.graph, input_layer, action_value, updated_action, loss


if __name__ == '__main__':
    Graph(action=10).get_graph()
