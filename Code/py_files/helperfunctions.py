def Dloss(y_train, y_pred, lossMethod="mse"):
    """
    Inputs:
    y_train: np.array, your Y training data set
    y_pred: np.array, your network predictions
    Desc:
    Vector of derivatives of costfunction given input
    """
    if lossMethod == "mse":
        return 2*(y_pred - y_train)


if __name__ == "__main__":
    pass
