import numpy as np
from layer import Layer


class Dense(Layer):
    def __init__(self, n, g):
        super(Dense, self).__init__(n, g)

    def Backpropagation(self,
                        y_train,
                        a_prev,
                        w_next,
                        delta
                        ):
        """
            Input:
            y_train: a numpy array, an element of your Y training set (labels)
            a_prev: a numpy array, activation output of a previous layer
            w_next: a numpy array, weights from the next layer
            delta: a numpy array, next delta from backpropagation
            at later layer
            Desc:
            Performs backpropagation algorithm for one hidden layer
        """
        w = w_next
        DzDw = a_prev
        DzDw = DzDw.reshape(-1, 1)
        ActivTerm = self.Dg(self.z)
        ActivTerm = ActivTerm.reshape(-1, 1)
        DzDz_prev = np.multiply(ActivTerm, w)
        delta = np.dot(DzDz_prev, delta)
        DcostDw = np.multiply(DzDw, delta.T)
        DcostDb = delta.T[0]
        self.dw = DcostDw
        self.db = DcostDb
        return delta
