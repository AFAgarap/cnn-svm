from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                strides=1, activation=tf.nn.relu, kernel_initializer="he_normal")
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                strides=1, activation=tf.nn.relu, kernel_initializer="he_normal")
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(units=1024,
                activation=tf.nn.relu, kernel_initializer="he_normal")
        self.dropout_layer = tf.keras.layers.Dropout(rate=5e-1)
        self.output_layer = tf.keras.layers.Dense(units=kwargs["num_classes"])

    def call(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = activations[len(activations) - 1]
        return logits

