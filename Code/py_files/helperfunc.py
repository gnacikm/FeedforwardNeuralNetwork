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
    return np.where(z>=0, 1, 0)

def ID(z):
    return z

def DID(z):
    return 1

# Dictionary of known activation functions
activation = {"sigmoid": sigmoid,
          "Dsigmoid": Dsigmoid,
          "tanh": np.tanh,
          "Dtanh": Dtanh,
          "id":   ID,
          "Did":  DID,
          "SoftPlus": SoftPlus,
          "DSoftPlus": sigmoid,
          "ReLu": ReLu,
          "DReLu": DReLu
          }

def costfunct(y_train, y_pred):
    """
    Inputs:
    y_train: np.array, your Y training data set
    y_pred: np.array, your network predictions
    Desc:
    Returns the sum of squares of errors
    """
    return np.sum((y_pred - y_train)**2)
    

def Dcost(y_train, y_pred):
    """
    Inputs:
    y_train: np.array, your Y training data set
    y_pred: np.array, your network predictions
    Desc:
    Vector of derivatives of costfunction given input
    """
    return 2*(y_pred - y_train)
 