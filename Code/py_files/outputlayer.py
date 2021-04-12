import numpy as np
from layer import Layer
from helperfunctions import Dloss


class OutputLayer(Layer):
    def __init__(self, n, g):
        super(OutputLayer, self).__init__(n, g)

    def Backpropagation(self,
                        y_train,
                        a_prev,
                        y_pred
                        ):
        """
            Input:
            y_train: a numpy array, an element of your Y training set (labels)
            a_prev: a numpy array, activation output of a previous layer
            y_pred:  a numpy array, the output of your network
            Desc:
            Performs backpropagation algorithm for the output layer
        """
        self.dcost_dz_last = Dloss(
            y_train,
            y_pred
            )*self.Dg(self.z)
        delta = self.dcost_dz_last[:]
        self.DzDweight_last = a_prev
        self.DzDweight_last = self.DzDweight_last.reshape(-1, 1)
        delta = delta.reshape(-1, 1)
        DcostDw = np.multiply(self.DzDweight_last, delta.T)
        DcostDb = delta.T[0]
        self.dw = DcostDw
        self.db = DcostDb
        return delta
