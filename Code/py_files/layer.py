import numpy as np
from activfunctions import activation


class Layer:
    def __init__(self, n, g):
        """
            Input:
            n: a tuple,
                (# of nodes in the previous layer,
                 # of  nodes in the current layer)
            g: a string or method, (activation function)
            Desc:
            weights: np.array, weights in the current layer, n[0] x n[1] matrix
            biases: np.array, biases in the current layer
            dw: np.array, derivative of cost function w.r.t.
            weights in the current layer
            db: np.array, derivative of cost function w.r.t.
            biases in the current layer
        """
        self.n = n[-1]
        if callable(g):
            self.g = g
            self.Dg = self.derivative(g)
        else:
            self.g = activation[g]
            self.Dg = activation["D{}".format(g)]
        self.weights = np.random.normal(0.0, 0.05, size=(n[0], n[1]))
        self.bias = np.zeros(n[-1])
        self.dw = np.zeros(shape=(n[0], n[1]))
        self.db = np.zeros(n[-1])
        self.z = np.zeros(n[-1])
        self.a = np.zeros(n[-1])

    def UpdateWeightsBiases(self, ModWeights, ModBias):
        """
            Input:
            ModWeights: np.array, same shape as self.weights
            ModBias: np.array, same shape as self.bias
            Desc:
            This function modifies self.weights and self.biases
            by subtracting ModWeights and ModBiases, respectively.
            Note that this is useful for Gradient Descent
            for adjusting weights/biases
        """
        self.weights -= ModWeights
        self.bias -= ModBias

    def Feedforward(self, aold):
        """
            Input:
            aold: a numpy array, activation output of previous layer
            Desc:
            Performs a feedforward algorithm and
            returns the activation output of the layer
        """
        self.z = np.matmul(aold.T, self.weights) + self.bias
        self.a = self.g(self.z)
        return self.a

    @staticmethod
    def derivative(fun,  h=1e-5):
        def df(z):
            return (fun(z + h) - fun(z))/h
        return df
