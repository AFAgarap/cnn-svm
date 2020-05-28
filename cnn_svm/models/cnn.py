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
"""Implementation of convolutional neural network"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(
            units=1024, activation=tf.nn.relu, kernel_initializer="he_normal"
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=5e-1)
        self.output_layer = tf.keras.layers.Dense(units=kwargs["num_classes"])
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = activations[len(activations) - 1]
        return logits

    def fit(self, data_loader, epochs):
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = self.epoch_train(self, data_loader)
            train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {train_loss[-1]:.6f}")

    @staticmethod
    def epoch_train(model, data_loader):
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            with tf.GradientTape() as tape:
                outputs = model(batch_features)
                train_loss = model.loss_fn(batch_labels, outputs)
            gradients = tape.gradient(train_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += train_loss
        epoch_loss = tf.reduce_mean(epoch_loss)
        return epoch_loss
