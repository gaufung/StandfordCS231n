import numpy as np

'''
linear softmax classifier
'''

def softmax_loss_naive(W, X, y, reg):
    '''
    Structured softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C * D array where C is the no. of classes and D is feature's count
    - X: D * N array of data where D is feature's count and N is batch sample point size
    - y: 1-dimensional array of length N with labels 0...K-1 for K class
    - reg: (float) regularization strength
    Returns: a tuple contianing
    - loss as single float
    - gradient with respect to weight W; an array of same shape of W
    '''
    dW = np.zeros(W.shape)
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = W.dot(X[:, i])
        dW[y[i], :] -= X[:, i].T
        #denominator = np.sum(np.exp(socres))
        logC = -1.0 * np.max(scores)
        denominator = np.sum(np.exp(scores + logC ))
        for j in range(num_classes):
            numerator = np.exp(scores[j] + logC)
            dW[j,:] += numerator/denominator * X[:,i].T
        loss += -1.0 * np.log(np.exp(scores[y[i]]+logC) / denominator)
    loss /= num_train
    dW /= num_train
    # regularization
    loss += 0.5 * reg * np.sum(W*W)
    dW += reg*W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    pass