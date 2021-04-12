import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def Dsigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2


def Dtanh(z):
    return 1/(np.cosh(z))**2


def SoftPlus(z):
    return np.log(1+np.exp(z))


def ReLu(z):
    return np.maximum(0, z)


def DReLu(z):
    return np.where(z >= 0, 1, 0)


# Dictionary of known activation functions
activation = {
    "sigmoid": sigmoid,
    "Dsigmoid": Dsigmoid,
    "tanh": np.tanh,
    "Dtanh": Dtanh,
    "SoftPlus": SoftPlus,
    "DSoftPlus": sigmoid,
    "ReLu": ReLu,
    "DReLu": DReLu
    }

if __name__ == "__main__":
    pass
