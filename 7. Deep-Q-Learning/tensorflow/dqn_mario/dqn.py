import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self):
        # First layer of weights
        """
        with tf.variable_scope(self.net_name):
            self._l_rate = tf.placeholder(tf.float32)
            self._X = tf.placeholder(tf.float32, [None, self.input_size],
                                     name="input_x")
            W1 = tf.get_variable("W1", shape=[self.input_size, 1000],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            # Second layer of Weights
            W2 = tf.get_variable("W2", shape=[1000, 1000],
                                 initializer=tf.contrib.layers.xavier_initializer())

            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))

            W3 = tf.get_variable("W3", shape=[1000, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())


            self._Qpred = tf.matmul(layer2, W3)
        """

        with tf.variable_scope(self.net_name):
                self._l_rate = tf.placeholder(tf.float32)
                self._X = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], self.input_size[2]], name="input_x")
                self._P = tf.placeholder(tf.float32, [None, 2, 6, 5], name="input_p")
            

     
     
                with tf.name_scope("VGG_Layer1"):
                    VGG_Layer1_1 = tf.layers.conv2d(self._X, filters=int(64), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer1_conv1')
                    VGG_Layer1_1 = tf.nn.relu(VGG_Layer1_1)
                    VGG_Layer1_2 = tf.layers.conv2d(VGG_Layer1_1, filters=int(64), kernel_size=[3, 3], strides=[2, 2],
                                                    padding='VALID', use_bias=False, name='VGG_Layer1_conv2')
                    VGG_Layer1_2 = tf.nn.relu(VGG_Layer1_2)
                     # shape (B, h, w, 64)->(B, h/2, w/2, 64)
                with tf.name_scope("VGG_Layer2"):
                    ########################################################################################################
                    VGG_Layer2_1 = tf.layers.conv2d(VGG_Layer1_2, filters=int(128), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer2_conv1')
                    VGG_Layer2_1 = tf.nn.relu(VGG_Layer2_1)  # shape (B, h/2, w/2, 64)->(B, h/2, w/2, 128)
                    ########################################################################################################
                    VGG_Layer2_2 = tf.layers.conv2d(VGG_Layer2_1, filters=int(128), kernel_size=[3, 3], strides=[2, 2],
                                                    padding='VALID', use_bias=False, name='VGG_Layer2_conv2')
                    VGG_Layer2_2 = tf.nn.relu(VGG_Layer2_2)  # shape (B, h/2, w/2, 128)->(B, h/4, w/4, 128)
                    ########################################################################################################

                with tf.name_scope("VGG_Layer3"):
                    ########################################################################################################
                    VGG_Layer3_1 = tf.layers.conv2d(VGG_Layer2_2, filters=int(256), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer3_conv1')
                    VGG_Layer3_1 = tf.nn.relu(VGG_Layer3_1)  # shape (B, h/4, w/4, 128)->(B, h/4, w/4, 256)
                    ########################################################################################################
                    VGG_Layer3_2 = tf.layers.conv2d(VGG_Layer3_1, filters=int(256), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer3_conv2')

                    VGG_Layer3_2 = tf.nn.relu(VGG_Layer3_2)  # shape (B, h/4, w/4, 256)->(B, h/4, w/4, 256)
                    ########################################################################################################
                    VGG_Layer3_3 = tf.layers.conv2d(VGG_Layer3_2, filters=int(256), kernel_size=[3, 3], strides=[2, 2],
                                                    padding='VALID', use_bias=False, name='VGG_Layer3_conv3')
                    VGG_Layer3_3 = tf.nn.relu(VGG_Layer3_3)  # shape (B, h/4, w/4, 256)->(B, h/8, w/8, 256)
                    ########################################################################################################

                    ########################################################################################################
                with tf.name_scope("VGG_Layer4"):
                    ########################################################################################################
                    VGG_Layer4_1 = tf.layers.conv2d(VGG_Layer3_3, filters=int(512), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer4_conv1')
                    VGG_Layer4_1 = tf.nn.relu(VGG_Layer4_1)  # shape (B, h/8, w/8, 256)->(B, h/8, w/8, 512)
                    ########################################################################################################
                    VGG_Layer4_2 = tf.layers.conv2d(VGG_Layer4_1, filters=int(512), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer4_conv2')
                    VGG_Layer4_2 = tf.nn.relu(VGG_Layer4_2)  # shape (B, h/8, w/8, 512)->(B, h/8, w/8, 512)
                    ########################################################################################################
                    VGG_Layer4_3 = tf.layers.conv2d(VGG_Layer4_2, filters=int(512), kernel_size=[3, 3], strides=[2, 2],
                                                    padding='VALID', use_bias=False, name='VGG_Layer4_conv3')
                    VGG_Layer4_3 = tf.nn.relu(VGG_Layer4_3)  # shape (B, h/8, w/8, 512)->(B, h/16, w/16, 512)
                    ########################################################################################################

                    ########################################################################################################
                with tf.name_scope("VGG_Layer5"):
                    ########################################################################################################
                    VGG_Layer5_1 = tf.layers.conv2d(VGG_Layer4_3, filters=int(512), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer5_conv1')
                    VGG_Layer5_1 = tf.nn.relu(VGG_Layer5_1)  # shape (B, h/16, w/16, 512)->(B, h/16, w/16, 512)
                    ########################################################################################################
                    VGG_Layer5_2 = tf.layers.conv2d(VGG_Layer5_1, filters=int(512), kernel_size=[3, 3],
                                                    padding='VALID', use_bias=False, name='VGG_Layer5_conv2')
                    VGG_Layer5_2 = tf.nn.relu(VGG_Layer5_2)  # shape (B, h/16, w/16, 512)->(B, h/16, w/16, 512)
                    ########################################################################################################
                    VGG_Layer5_3 = tf.layers.conv2d(VGG_Layer5_2, filters=int(512), kernel_size=[3, 3], strides=[2, 2],
                                                    padding='VALID', use_bias=False, name='VGG_Layer5_conv3')
                    VGG_Layer5_3 = tf.nn.relu(VGG_Layer5_3)  # shape (B, h/16, w/16, 512)->(B, h/32, w/32, 512)
                    ########################################################################################################

                    ########################################################################################################
                with tf.name_scope("VGG_Qpred"):
                    VGG_Layer6_1 = tf.layers.conv2d(VGG_Layer5_3, filters=100, kernel_size=[2, 3], strides=[1, 1], padding='VALID')
                    VGG_Layer6_1 = tf.nn.relu(VGG_Layer6_1)
                    VGG_Layer6_1 = tf.contrib.layers.flatten(VGG_Layer6_1)
                    JOYSTICK_Layer = tf.contrib.layers.flatten(self._P)
                    VGG_Layer6_2 = tf.concat([VGG_Layer6_1, JOYSTICK_Layer], axis=1)
                    VGG_Layer6_2 = tf.layers.dense(VGG_Layer6_2, units=100, use_bias=False)
                    VGG_Layer6_2 = tf.nn.relu(VGG_Layer6_2)
                    VGG_Layer6_3 = tf.layers.dense(VGG_Layer6_2, units=50, use_bias=False)
                    VGG_Layer6_3 = tf.nn.relu(VGG_Layer6_3)
                    self._Qpred = tf.layers.dense(VGG_Layer6_3, units=self.output_size, use_bias=False)



            # We need to define the parts of the network needed for learning a policy

        self._Y = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)

            # Loss function

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
            # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=self._l_rate).minimize(self._loss)

    def predict(self, state, action_seq):
        #x = np.reshape(state, [1, self.input_size])
        x = np.reshape(state, [1, self.input_size[0], self.input_size[1], self.input_size[2]])
        action_seq = np.reshape(action_seq, [1, 2, 6, 5])
        return self.session.run(self._Qpred, feed_dict={self._X: x, self._P: action_seq})



    def update(self, x_stack, y_stack, action_seq, l_rate = 1e-5):
        #x_stack = np.reshape(x_stack, (-1, self.input_size))
        x_stack = np.reshape(x_stack, (-1, self.input_size[0], self.input_size[1], self.input_size[2]))
        action_seq = np.reshape(action_seq, [-1, 2, 6, 5])
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack, self._P: action_seq,  self._l_rate: l_rate})
