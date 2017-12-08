# Copyright 2017 Abien Fred Agarap

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2 Convolutional Layers with Max Pooling CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys


class CNN:

    def __init__(self, alpha, batch_size, num_classes, sequence_length):
        """Initializes the CNN-Softmax model

        :param alpha: The learning rate to be used by the model.
        :param batch_size: The number of batches to use for training/validation/testing.
        :param num_classes: The number of classes in the dataset.
        :param sequence_length: The number of features in the dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        def __graph__():
            # [BATCH_SIZE, 784]
            x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length], name='x_input')

            # [BATCH_SIZE, 10]
            y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='actual_label')

            # First convolutional layer
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])

            x_image = tf.reshape(x, [-1, 28, 28, 1])

            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            # Second convolutional layer
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)

            # Fully-connected layer (Dense Layer)
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Dropout
            # For avoiding overfitting
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # Readout layer
            W_fc2 = self.weight_variable([1024, num_classes])
            b_fc2 = self.bias_variable([num_classes])
            
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            with tf.name_scope('softmax'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
            tf.summary.scalar('loss', loss)

            train_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

            y_conv = tf.identity(tf.nn.softmax(y_conv), name='prediction')
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.x = x
            self.y = y
            self.keep_prob = keep_prob
            self.y_conv = y_conv
            self.loss = loss
            self.train_op = train_op
            self.accuracy_op = accuracy_op

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, epochs, train_data, test_data):
        """Trains the initialized model.

        :param epochs: The number of passes through the entire dataset.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :return: None
        """
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for index in range(epochs):
                # train by batch
                batch_x, batch_y = train_data.next_batch(self.batch_size)

                # input dictionary with dropout of 50%
                feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5}
                
                # run the train op
                _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                
                # every 100th step and at 0,
                if index % 100 == 0:
                    feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0}
                    
                    # get the accuracy of training
                    train_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
                    
                    # display the training accuracy
                    print('step: {}, training accuracy : {}, training loss : {}'.format(index, train_accuracy, loss))

            y_test = test_data.labels
            y_test[y_test == 0] = -1
            feed_dict = {self.x: test_data.images, self.y: y_test, self.keep_prob: 1.0}

            test_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)

            print('Test Accuracy: {}'.format(test_accuracy))

    @staticmethod
    def weight_variable(shape):
        """Returns a weight matrix consisting of arbitrary values.

        :param shape: The shape of the weight matrix to create.
        :return: The weight matrix consisting of arbitrary values.
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Returns a bias matrix consisting of 0.1 values.

        :param shape: The shape of the bias matrix to create.
        :return: The bias matrix consisting of 0.1 values.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        """Produces a convolutional layer that filters an image subregion

        :param x: The layer input.
        :param W: The size of the layer filter.
        :return: Returns a convolutional layer.
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """Downnsamples the image based on convolutional layer

        :param x: The input to downsample.
        :return: Downsampled input.
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def parse_args():
    parser = argparse.ArgumentParser(description='2 Convolutional Layer with Max Pooling for MNIST Classification')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                             help='path of the MNIST dataset')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    mnist = input_data.read_data_sets(args.dataset, one_hot=True)
    num_classes = mnist.train.labels.shape[1]
    sequence_length = mnist.train.images.shape[1]
    model = CNN(alpha=1e-4, batch_size=128, num_classes=num_classes, sequence_length=sequence_length)
    model.train(epochs=2000, train_data=mnist.train, test_data=mnist.test)
