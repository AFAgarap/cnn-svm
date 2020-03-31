# Copyright 2017-2020 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""2 Convolutional Layers with Max Pooling CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import os
import tensorflow as tf
import time
import sys


class CNNSVM:
    def __init__(self, alpha, batch_size, num_classes, num_features, penalty_parameter):
        """Initializes the CNN-SVM model

        :param alpha: The learning rate to be used by the model.
        :param batch_size: The number of batches to use for training/validation/testing.
        :param num_classes: The number of classes in the dataset.
        :param num_features: The number of features in the dataset.
        :param penalty_parameter: The SVM C penalty parameter.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.name = "CNN-SVM"
        self.num_classes = num_classes
        self.num_features = num_features
        self.penalty_parameter = penalty_parameter

        def __graph__():

            with tf.name_scope("input"):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, num_features], name="x_input"
                )

                # [BATCH_SIZE, NUM_CLASSES]
                y_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, num_classes], name="actual_label"
                )

            # First convolutional layer
            first_conv_weight = self.weight_variable([5, 5, 1, 32])
            first_conv_bias = self.bias_variable([32])

            input_image = tf.reshape(x_input, [-1, 28, 28, 1])

            first_conv_activation = tf.nn.relu(
                self.conv2d(input_image, first_conv_weight) + first_conv_bias
            )
            first_conv_pool = self.max_pool_2x2(first_conv_activation)

            # Second convolutional layer
            second_conv_weight = self.weight_variable([5, 5, 32, 64])
            second_conv_bias = self.bias_variable([64])

            second_conv_activation = tf.nn.relu(
                self.conv2d(first_conv_pool, second_conv_weight) + second_conv_bias
            )
            second_conv_pool = self.max_pool_2x2(second_conv_activation)

            # Fully-connected layer (Dense Layer)
            dense_layer_weight = self.weight_variable([7 * 7 * 64, 1024])
            dense_layer_bias = self.bias_variable([1024])

            second_conv_pool_flatten = tf.reshape(second_conv_pool, [-1, 7 * 7 * 64])
            dense_layer_activation = tf.nn.relu(
                tf.matmul(second_conv_pool_flatten, dense_layer_weight)
                + dense_layer_bias
            )

            # Dropout, to avoid over-fitting
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(dense_layer_activation, keep_prob)

            # Readout layer
            readout_weight = self.weight_variable([1024, num_classes])
            readout_bias = self.bias_variable([num_classes])

            output = tf.matmul(h_fc1_drop, readout_weight) + readout_bias

            with tf.name_scope("svm"):
                regularization_loss = tf.reduce_mean(tf.square(readout_weight))
                hinge_loss = tf.reduce_mean(
                    tf.square(
                        tf.maximum(
                            tf.zeros([batch_size, num_classes]), 1 - y_input * output
                        )
                    )
                )
                with tf.name_scope("loss"):
                    loss = regularization_loss + penalty_parameter * hinge_loss
            tf.summary.scalar("loss", loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

            with tf.name_scope("accuracy"):
                output = tf.identity(tf.sign(output), name="prediction")
                correct_prediction = tf.equal(
                    tf.argmax(output, 1), tf.argmax(y_input, 1)
                )
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.keep_prob = keep_prob
            self.output = output
            self.loss = loss
            self.optimizer = optimizer
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(self, checkpoint_path, epochs, log_path, train_data, test_data):
        """Trains the initialized model.

        :param checkpoint_path: The path where to save the trained model.
        :param epochs: The number of passes through the entire dataset.
        :param log_path: The path where to save the TensorBoard logs.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :return: None
        """

        if not os.path.exists(path=log_path):
            os.mkdir(log_path)

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(checkpoint_path)

        saver = tf.train.Saver(max_to_keep=4)

        init = tf.global_variables_initializer()

        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(
            logdir=log_path + timestamp + "-training", graph=tf.get_default_graph()
        )

        with tf.Session() as sess:
            sess.run(init)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(
                    checkpoint.model_checkpoint_path + ".meta"
                )
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

            for index in range(epochs):
                # train by batch
                batch_features, batch_labels = train_data.next_batch(self.batch_size)
                batch_labels[batch_labels == 0] = -1

                # input dictionary with dropout of 50%
                feed_dict = {
                    self.x_input: batch_features,
                    self.y_input: batch_labels,
                    self.keep_prob: 0.5,
                }

                # run the train op
                summary, _, loss = sess.run(
                    [self.merged, self.optimizer, self.loss], feed_dict=feed_dict
                )

                # every 100th step and at 0,
                if index % 100 == 0:
                    feed_dict = {
                        self.x_input: batch_features,
                        self.y_input: batch_labels,
                        self.keep_prob: 1.0,
                    }

                    # get the accuracy of training
                    train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

                    # display the training accuracy
                    print(
                        "step: {}, training accuracy : {}, training loss : {}".format(
                            index, train_accuracy, loss
                        )
                    )

                    train_writer.add_summary(summary=summary, global_step=index)

                    saver.save(
                        sess,
                        save_path=os.path.join(checkpoint_path, self.name),
                        global_step=index,
                    )

            test_features = test_data.images
            test_labels = test_data.labels
            test_labels[test_labels == 0] = -1

            feed_dict = {
                self.x_input: test_features,
                self.y_input: test_labels,
                self.keep_prob: 1.0,
            }

            test_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

            print("Test Accuracy: {}".format(test_accuracy))

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
    def conv2d(features, weight):
        """Produces a convolutional layer that filters an image subregion

        :param features: The layer input.
        :param weight: The size of the layer filter.
        :return: Returns a convolutional layer.
        """
        return tf.nn.conv2d(features, weight, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(features):
        """Downnsamples the image based on convolutional layer

        :param features: The input to downsample.
        :return: Downsampled input.
        """
        return tf.nn.max_pool(
            features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
