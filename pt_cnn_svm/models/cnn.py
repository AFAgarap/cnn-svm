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
"""Implementation of convolutional network in PyTorch"""
import torch


class CNN(torch.nn.Module):
    """
    A convolutional neural network that optimizes
    softmax cross entropy using Adam optimizer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_classes: int = 10,
        model_device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-4,
    ):
        """
        Constructs a convolutional neural network classifier.

        Parameters
        ----------
        input_dim: int
            The dimensionality of the input feature channel.
        num_classes: int
            The number of classes in the dataset.
        model_device: torch.device
            The device to use for model computations.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=64,
                    kernel_size=8,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=(128 * 5 * 5), out_features=2048),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2048, out_features=2048),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2048, out_features=512),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=512, out_features=num_classes),
            ]
        )
        self.model_device = model_device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.model_device)
        self.train_loss = []

    def forward(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        logits : torch.Tensor
            The model output.
        """
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = activations[len(activations) - 1]
        return logits

    def predict(self, features, return_likelihoods=False):
        """
        Returns model classifications

        Parameters
        ----------
        features: torch.Tensor
            The input features to classify.
        return_likelihoods: bool
            Whether to return classes with likelihoods or not.

        Returns
        -------
        predictions: torch.Tensor
            The class likelihood output by the model.
        classes: torch.Tensor
            The class prediction by the model.
        """
        outputs = self.forward(features)
        predictions, classes = torch.max(outputs.data, dim=1)
        return (predictions, classes) if return_likelihoods else classes

    def fit(self, data_loader, epochs):
        """
        Trains the cnn model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        """
        self.to(self.model_device)
        for epoch in range(epochs):
            epoch_loss = self.epoch_train(self, data_loader)
            if "cuda" in self.model_device.type:
                torch.cuda.empty_cache()
            self.train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}")

    @staticmethod
    def epoch_train(model, data_loader):
        """
        Trains a model for one epoch.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.

        Returns
        -------
        epoch_loss : float
            The epoch loss.
        """
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(model.model_device)
            batch_labels = batch_labels.to(model.model_device)
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_labels)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()

        epoch_loss /= len(data_loader)

        return epoch_loss
