import numpy as np
'''
linear svm implementation
'''
def svm_loss_naive(W, X, y, reg, delta=1.0):
    '''
    Structured SVM loss function, navie implementation (with loops)
    Inputs:
    - W: C * D array where C is the no. of classes and D is feature's count
    - X: D * N array of data where D is feature's count and N is batch sample point size
    - y: 1-dimensional array of length N with labels 0...K-1 for K class
    - reg: (float) regularization strength
    - delta: the margin
    Returns: a tuple contianing
    - loss as single float
    - gradient with respect to weight W; an array of same shape of W
    '''
    dW = np.zeros(W.shape)
    # compute the loss and gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                # add gradient 
                dW[y[i], :] -= X[:, i].T
                dW[j, :] += X[:, i].T
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW += reg*W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg, delta=1.0):
    '''
    Structured SVM loss function, vectorized implementation
    Inputs:
    - W: C * D array where C is the no. of classes and D is feature's count
    - X: D * N array of data where D is feature's count and N is batch sample point size
    - y: 1-dimensional array of length N with labels 0...K-1 for K class
    - reg: (float) regularization strength
    - delta: the margin
    Returns: a tuple contianing
    - loss as single float
    - gradient with respect to weight W; an array of same shape of W
    '''
    loss = 0.0
    num_train = X.shape[1]
    dW = np.zeros(W.shape)
    scores = W.dot(X)
    Loss = scores - scores[y,np.arange(num_train)] + delta
    Loss[y, np.arange(num_train)] = 0
    margin_bool = Loss > 0
    Loss = np.sum(Loss*margin_bool, axis=0) - delta
    regularization = 0.5 * reg * np.sum(W*W)
    loss = np.sum(Loss) / num_train + regularization
    
    
    # loss = 0.0
    # num_train = X.shape[1]
    # dW = np.zeros(W.shape)
    # Loss = W.dot(X) - (W.dot(X))[y, np.arange(num_train)] + delta
    # Bool = loss > 0 
    # loss = np.sum(Loss * Bool, axis=0) - delta
    # Regularization = 0.5 * reg * np.sum(W*W)
    # Loss = np.sum(Loss) / num_train + Regularization
    # Bool = Bool * np.ones(Loss.shape)
    # Bool[[y, np.arange(num_train)]] = -(np.sum(Bool, axis=0) - delta)
    # dW = Bool.dot(X.T) / num_train
    # dW += reg*W
    # return loss, dW
    # loss = 0.0
    # dW = np.zeros(W.shape)
    # D = X.shape[0]
    # num_classes = W.shape[0]
    # num_train = X.shape[1]
    # scores = W.dot(X)
    # correct_scores = scores[y, np.arange(num_train)]
    # mat = scores - correct_scores + delta
    # mat[y, np.arange(num_train)] = 0
    # thresh = np.maximum(np.zeros((num_classes, num_train)), mat)
    # loss = np.sum(thresh)
    # loss /= num_train
    # loss += 0.5 * reg * np.sum(W*W)
    # binary = thresh
    # binary[thresh>0] = delta
    # col_sum = np.sum(binary, axis=0)
    # binary[y, range(num_train)] = -col_sum[range(num_train)]
    # dW = np.dot(binary, X.T)

  # Divide
    # dW /= num_train

  # Regularize
    # dW += reg*W

    # return loss, dW
    pass