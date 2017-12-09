An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification
===

![](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)]()

*This project was inspired by Y. Tang's [Deep Learning using Support Vector Machines](https://arxiv.org/abs/1306.0239)
(2013).*

## Abstract 

Convolutional Neural Networks (CNNs) are similar to "ordinary" neural networks in the sense that they are made up of
hidden layers consisting neurons with "learnable" parameters. These neurons receive inputs, performs a dot product, and
then follows it with a non-linearity. The whole network expresses the mapping between raw image pixels and class scores.
Conventionally, the Softmax function is the classifier used at the last layer of this network.

There have been studies ([Alalshekmubarak & Smith, 2013](http://ieeexplore.ieee.org/abstract/document/6544391/); 
[Agarap, 2017](http://arxiv.org/abs/1709.03082); [Tang, 2013](https://arxiv.org/abs/1306.0239)) on the usage of a
different classifier than Softmax for neural networks. The aforementioned studies introduce the usage of support vector
machine (SVM) in an artificial neural network architecture. This project is yet another take on the subject, and is
inspired by [[Tang, 2017](https://arxiv.org/abs/1306.0239)].

## Usage

First, clone the project.
```bash
git clone https://github.com/AFAgarap/cnn-svm.git/
```

Run the `setup.sh` to ensure that the pre-requisite libraries are installed in the environment.
```bash
sudo chmod +x
./setup.sh
```

Program parameters.
```bash
usage: main.py [-h] -m MODEL -d DATASET [-p PENALTY_PARAMETER] -c
               CHECKPOINT_PATH -l LOG_PATH

CNN & CNN-SVM for Image Classification

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m MODEL, --model MODEL
                        [1] CNN-Softmax, [2] CNN-SVM
  -d DATASET, --dataset DATASET
                        path of the MNIST dataset
  -p PENALTY_PARAMETER, --penalty_parameter PENALTY_PARAMETER
                        the SVM C penalty parameter
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        path where to save the trained model
  -l LOG_PATH, --log_path LOG_PATH
                        path where to save the TensorBoard logs
```

Then, go to the repository's directory, and run the `main.py` module as per the desired parameters.
```bash
cd cnn-svm
python3 main.py --model 2 --dataset ./MNIST_data --penalty_parameter 1 --checkpoint_path ./checkpoint --log_path ./logs
```

## Results

The hyperparameters used in this project were manually assigned, and not through optimization.

|Hyperparameters|CNN-Softmax|CNN-SVM|
|---------------|-----------|-------|
|Batch size|128|128|
|Epochs|10000|10000|
|Learning rate|1e-3|1e-3|
|SVM C|N/A|1|

The experiments were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM,
and NVIDIA GeForce GTX 960M 4GB DDR5 GPU.

![](figures/accuracy-loss.png)

**Figure 1. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on image classification using
[MNIST](http://yann.lecun.com/exdb/mnist/).**

The orange plot refers to the training accuracy and loss of CNN-Softmax, with a test accuracy of 99.22999739646912%.
On the other hand, the blue plot refers to the training accuracy and loss of CNN-SVM, with a test accuracy of
99.04000163078308%. The results do not corroborate the findings of [Tang (2017)](https://arxiv.org/abs/1306.0239)
for [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) classification. This may be attributed to the fact
that no data preprocessing nor dimensionality reduction was done on the dataset for this project.


## Citation
```
```

## License
```
Copyright 2017 Abien Fred Agarap

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
