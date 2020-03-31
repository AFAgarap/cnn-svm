# Copyright 2017-2020 Abien Fred Agarap

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the CNN classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import argparse
from model.cnn_softmax import CNN
from model.cnn_svm import CNNSVM
from tensorflow.examples.tutorials.mnist import input_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN & CNN-SVM for Image Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-m", "--model", required=True, type=str, help="[1] CNN-Softmax, [2] CNN-SVM"
    )
    group.add_argument(
        "-d", "--dataset", required=True, type=str, help="path of the MNIST dataset"
    )
    group.add_argument(
        "-p",
        "--penalty_parameter",
        required=False,
        type=int,
        help="the SVM C penalty parameter",
    )
    group.add_argument(
        "-c",
        "--checkpoint_path",
        required=True,
        type=str,
        help="path where to save the trained model",
    )
    group.add_argument(
        "-l",
        "--log_path",
        required=True,
        type=str,
        help="path where to save the TensorBoard logs",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()

    mnist = input_data.read_data_sets(args.dataset, one_hot=True)
    num_classes = mnist.train.labels.shape[1]
    sequence_length = mnist.train.images.shape[1]
    model_choice = args.model

    assert (
        model_choice == "1" or model_choice == "2"
    ), "Invalid choice: Choose between 1 and 2 only."

    if model_choice == "1":
        model = CNN(
            alpha=1e-3,
            batch_size=128,
            num_classes=num_classes,
            num_features=sequence_length,
        )
        model.train(
            checkpoint_path=args.checkpoint_path,
            epochs=10000,
            log_path=args.log_path,
            train_data=mnist.train,
            test_data=mnist.test,
        )
    elif model_choice == "2":
        model = CNNSVM(
            alpha=1e-3,
            batch_size=128,
            num_classes=num_classes,
            num_features=sequence_length,
            penalty_parameter=args.penalty_parameter,
        )
        model.train(
            checkpoint_path=args.checkpoint_path,
            epochs=10000,
            log_path=args.log_path,
            train_data=mnist.train,
            test_data=mnist.test,
        )
