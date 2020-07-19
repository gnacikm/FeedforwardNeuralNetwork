# Theory and Code for Feedforward Neural Networks

This repository has an educational character; the existing neural networks Python libraries such as Keras are much more efficient. 

We hope that the readers who wish to start their journey with neural networks will find this repository useful. 

The structure of this repository is the following:

* **Code** - contains all the Python code for the Feedforward Neural Networks and Examples of application, in particular, on MNIST data set (handwritten digits).
* **Data** - contains data sets.
* **Notes** - contains the notes (.pdf and .tex files) that should help you implementing your Feedforward Neural Network from scratch (as presented in **Code**). It may be more accessible for the readers with mathematical background. 

In the notebook **ExamplesNN.ipynb** we demonstrate how our neural network performs on various data sets, including MNIST and Fashion MNIST. 
In order to run **ExamplesNN.ipynb** please first generate MNIST data set and Fashion MNIST (we get it via Keras <code>from keras.datasets import mnist, fashion_mnist</code>) via running all cells from the Notebooks: 
* **PreparingDataMNIST.ipynb**;
* **PreparingDataFashionMNIST.ipynb**.

The Class NeuralNetwork is written in the file **ANNfeedforward.py**. The file **helperfunc.py** contains activation functions.

For any inquiries please  <a href = "mailto: michal.gnacik@port.ac.uk">contact me via email</a>.