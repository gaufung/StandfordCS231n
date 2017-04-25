import numpy as np
from linear_softmax import softmax_loss_naive
from linear_svm import svm_loss_naive

class LinearClassifier(object):
    def __init__(self):
        self.W = None
    def fit(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        '''
        train the linear classifier using stochasitc gradient descent

        Inputs:
        - X: D * N array of training data
        - y: 1-dimensional array of length N with labels from 0 to K-1, for K classes
        - learning_rate: (float)learning rate for optimization
        - reg: (floating)regularization strength
        - batch_size: (integer) number of training example to use at each step
        - verbose: (boolean) If true, print progress during optimization.

        Output:
        A list of containing the value of the loss function at each training iterations.
        '''
        dim, num_train = X.shape
        num_class = np.max(y) + 1
        if self.W is None:
            # initialize the weight W
            self.W = np.random.randn(num_class, dim) * 0.001
        # run Stochastic gradient descent to optimiz W 
        loss_histroy = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[:,indices]
            y_batch = y[indices]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_histroy.append(loss)
            self.W = self.W - learning_rate*grad
            if verbose and it % 100 == 0:
                print ('iteration', it, '/', num_iters,': loss %f'%loss)
        return loss_histroy
    def loss(self, X_batch, y_batch, reg):
        '''
        Compute the loss fucntion and its derivative
        Subclasses will implement this.

        Inputs:
        - X_batch: D * N array of data; each column is a data point
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - reg: (float) regularization stregth

        Returns: A tuple containing:
        - loss as single float
        - gradient with respect to self.w; an array of the same as W
        '''
        pass
    
    def predict(self, X):
        '''
        Use the trained weight of this linear classifier to predict labels for data points
        Inputs:
        - X: D * N array of training data. Each column is a D-Dimensional point.
        Returns:
        - y_pred: Predicated labels for the data in X y_pred is a 1-dimensional array of 
        length N, and each element is an integer giving the predicted class.
        '''
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(self.W.dot(X), axis=0)
        return y_pred

class LinearSVM(LinearClassifier):
    '''
    the subclass that uses the multiclasses svm loss function
    '''
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_naive(self.W, X_batch, y_batch, reg)

class LinearSoftmax(LinearClassifier):
    '''
    the subclass that uses the multiclasses softmax loss function
    '''
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_naive(self.W, X_batch, y_batch, reg)