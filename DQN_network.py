import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")


#             self.fc1 = tf.layers.dense(inputs = self.inputs_,
#                                   units = 101,
#                                   activation = tf.nn.relu,
#                                 name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.inputs_,
                                  units = 32,
                                  activation = tf.nn.relu,
                                name="fc2")
            self.fc3 = tf.layers.dense(inputs = self.fc2,
                                  units = 16,
                                  activation = tf.nn.relu,
                                name="fc3")


            self.output = tf.layers.dense(inputs = self.fc3,
                                          units = self.action_size,
                                        activation=None)


            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)


            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
