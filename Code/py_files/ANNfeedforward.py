import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from activfunctions import activation



class Layer:
    
    def __init__(self, n, g):
        """
            Input:
            n: a tuple, 
                (# of nodes in the previous layer, # of  nodes in the current layer)
            g: a string or method, (activation function)
            Desc:
            weights: np.array, weights in the current layer, n[0] x n[1] matrix
            biases: np.array, biases in the current layer
            dw: np.array, derivative of cost function w.r.t. weights in the current layer
            db: np.array, derivative of cost function w.r.t biases in the current layer
        """
        self.n = n[-1]
    
        if callable(g):
            self.g = g
            self.Dg = self.derivative(g)
        else:
            self.g = activation[g]
            self.Dg = activation["D{}".format(g)]
        #self.weights = np.random.uniform(low = -0.05, high=0.05, size=(n[0],n[1]))
        self.weights = np.random.normal(0.0, 0.05, size=(n[0],n[1]))
        self.bias = np.zeros(n[-1])
        self.dw = np.zeros(shape=(n[0],n[1]))
        self.db =  np.zeros(n[-1])
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
            Note that this is useful for Gradient Descent for adjusting weights/biases
        """
        self.weights -= ModWeights
        self.bias  -= ModBias
        
    def Feedforward(self, aold):
        """
            Input:
            aold: a numpy array, activation output of previous layer
            Desc:
            Performs a feedforward algorithm and 
            returns the activation output of the layer
        """
        self.z= aold.T@self.weights+self.bias
        self.a = self.g(self.z)
        return self.a
        
        
    @staticmethod
    def derivative(fun,  h = 1e-5):
        def df(z):
            return (fun(z+h)-fun(z))/h
        return df
    
        
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
            delta: a numpy array, next delta from backpropagation at later layer
            Desc:
            Performs backpropagation algorithm for one hidden layer
        """
        w =  w_next
        DzDw =  a_prev
        DzDw = DzDw.reshape(-1, 1)
        ActivTerm = self.Dg(self.z)
        ActivTerm = ActivTerm.reshape(-1, 1)
        DzDz_prev = np.multiply(ActivTerm, w)
        delta = np.dot(DzDz_prev, delta)
        DcostDw = np.multiply(DzDw, delta.T)
        DcostDb = delta.T[0]
        self.dw =  DcostDw
        self.db = DcostDb
        return delta
            
       
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
        self.dcost_dz_last = FeedForwardANN.Dloss(y_train, y_pred)*self.Dg(self.z)
        delta= self.dcost_dz_last[:]
        self.DzDweight_last = a_prev
        self.DzDweight_last = self.DzDweight_last.reshape(-1, 1)
        delta = delta.reshape(-1, 1)
        DcostDw = np.multiply(self.DzDweight_last,delta.T) 
        DcostDb = delta.T[0]
        self.dw = DcostDw 
        self.db = DcostDb    
        return delta
    
         
class FeedForwardANN:
    
    def __init__(self, shape, ActivFun):
        """
            Input:
            shape:  a tuple, the length represents the number of layers,
                    its elements represent the number of nodes/neurons 
                    at the corresponding layer
            ActivFun: a list, elements of a name of an activation function or an activation function itself
            Desc:
            Inside the class we create the list of all hidden layers + output layer, an elements of this list
            are objects of Layer class
        
        """
        self.shape = np.array(shape)
        self.size = self.shape.size
        
        if len(ActivFun)== 1:
            self.ActivFun = [ActivFun[0] for k in range(self.size-1)]
        elif len(ActivFun) == 2:
            self.ActivFun = [ActivFun[0] for k in range(self.size-2)] + [ActivFun[-1]]
        else:
            assert len(ActivFun) == self.size-1
            self.ActivFun = ActivFun
        
        
        self.layers = [
            Dense((shape[k-1],shape[k]),
                  self.ActivFun[k-1])
            for k in range(1,self.size-1)
        ] + [OutputLayer((shape[-2], shape[-1]), self.ActivFun[-1])]
        
   
  
    def feedforward(self, x_input):
        """
            Input:
            x_input: a numpy array
            Desc:
            Performs a feedforward algorithm and returns the network output $\hat{y}$
        """
        self.a0 = x_input
        aold = self.a0
        for k in range(self.size-1):
            aold = self.layers[k].Feedforward(aold)
        return aold
 

    def backpropagation(self, x_train, y_train):
        """
            Input:
            x_train: a numpy array, an element of your X training set (features)
            y_train: a numpy array, an element of your Y training set (labels)
            Desc:
            Performs backpropagation algorithm and stores the values of derivatives of 
            the cost function with respect to weights and biases. 
        """
        y_pred = self.feedforward(x_train)
        if self.size == 2:
            a_prev = self.a0
        else:
            a_prev = self.layers[-2].a
        
        delta = self.layers[-1].Backpropagation(y_train, 
                                       a_prev, 
                                       y_pred = y_pred 
                                       )
        for k in range(self.size-3, -1, -1):
            if k == 0:
                a_prev = self.a0
            else:
                a_prev = self.layers[k-1].a
            w_next = self.layers[k+1].weights
            delta = self.layers[k].Backpropagation(y_train, 
                                                  a_prev,
                                                  w_next = w_next, 
                                                  delta = delta
                                                  )
                 
    def train(self, x_train,
              y_train,
              epochs, 
              optimizer = "sgd",
              minibatch_size=50,
              learning_rate=0.1,
              myseed = 0,
              mnist = True
              ):
        """
            Input:
                x_train: a numpy array - full X training data set
                y_train: a numpy array - full Y training data set
                epochs: int - the number of epochs that you wish to run
                optimizer: str - specifying your learning algorithm 
                minibatch_size: int, the size of your minibatch for learning algorithm 
                learning_rate: float, the learning rate for learning algorithm 
                myseed: int, fixing the seed for np.random
            Desc:
            Perform the training of your network, prepares the mini-batches for the SGD
        
        """
        np.random.seed(myseed)
        ind = np.arange(x_train.shape[0])
        for j in tqdm(range(epochs), desc = "epochs"):
            ind_choice = np.random.choice(ind, size=x_train.shape[0], replace=False)
            x_train_copy = x_train[ind_choice]
            y_train_copy = y_train[ind_choice]
            batchx = np.array( 
                [x_train_copy[i: i+ minibatch_size] for i in range(0,x_train.shape[0], minibatch_size)])
            batchy = np.array( 
                [y_train_copy[i: i+ minibatch_size] for i in range(0,x_train.shape[0], minibatch_size)])

            for x, y in zip(batchx, batchy):
                self.update(x, y, learning_rate, optimizer=optimizer)
            if mnist == True:
                print("loss: {}, accuracy: {}".format(self.loss(x_train, y_train), self.score(x_train, y_train)))
          
    def update(self, x, y, learning_rate = 0.1, optimizer = "sgd"): 
        """
            Input:
                x: a numpy array - a mini-batch from X training set
                y: a numpy array - a mini-batch from Y training set
            Desc:
                Performs SGD to adjust your weights and biases
        """
        samp_size = x.shape[0]
        for x_samp, y_samp in zip(x, y):
            self.backpropagation(x_samp, y_samp)
            for j in range(self.size-1):
                if optimizer == "sgd":
                    ModWeights = (learning_rate/samp_size)*self.layers[j].dw
                    ModBias = (learning_rate/samp_size)*self.layers[j].db
                self.layers[j].UpdateWeightsBiases(ModWeights, ModBias) 
                
    def loss(self, x_input, y_labels, lossMethod = "mse"):
        """
        Input:
        x_input: a numpy array, your X training/test set
        y_labels: a numpy array, your Y training/test set (matching X)
        Desc:
        Returns the average cost of your network
        """
        y_pred = np.array([self.feedforward(x) for x in  x_input])
        if lossMethod == "mse":
            lossValue = np.array([(pred -  y_item)**2 for pred, y_item in zip(y_pred, y_labels)]).mean()
        return lossValue 
    
    
    def Dloss(y_train, y_pred, lossMethod = "mse"):
        """
        Inputs:
        y_train: np.array, your Y training data set
        y_pred: np.array, your network predictions
        Desc:
        Vector of derivatives of costfunction given input
        """
        if lossMethod == "mse":
            return 2*(y_pred - y_train)
    
    def score(self, x_test, y_test, mnist = True):
        """
            Input:
            x_test: a numpy array - your X training set
            Y_test: a numpy array - your Y training set
            Desc:
            This gives the % accuracy of your predictions for the MNIST dataset
        
        """
        assert mnist == True
        preds = np.array([self.feedforward(x) for x in x_test])
        preds_format = preds.argmax(axis = 1)
        label_format = y_test.argmax(axis = 1)
        correct_preds = label_format[label_format == preds_format].size
        return correct_preds/x_test.shape[0]
    
 
        
    def plotnetwork(self, save_file=False):
        """
        Desc: This function produces a NetworkX plot of your network
        """
        assert self.shape.size <= 5
        assert self.shape.max() <= 11
        self.G = nx.DiGraph()
        layer = self.shape
        i = 0
        m = layer.max()
        mid = m//2
        layer_names = [[] for k in range(layer.size)]
        for k, nodes in enumerate(layer):
            y = np.linspace(mid - nodes, mid + nodes, nodes+1)
            x = np.linspace(0, 100, 10)
            for j in range(nodes):
                self.G.add_node(i, pos = (x[k], y[j]))
                layer_names[k].append(i)
                i+=1

        for k, item in enumerate(layer_names):
            if k == len(layer_names)-1:
                break
            for node1 in item:
                for node2 in layer_names[k+1]:
                    self.G.add_edge(node1, node2)

        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos=pos)
        if save_file != False:
            plt.savefig(save_file)



