import numpy as np
import pickle
import os
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as fo:
        datadict = pickle.load(fo, encoding="bytes")
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10():
    '''
    load all the cifar
    '''
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join('data', 'data_batch_%d' % (b, ))
        X, y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, y
    Xte, Yte = load_CIFAR_batch(os.path.join('data', 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    Xtr, Ytr, Xte, Yte = load_CIFAR10()
    print(Xtr.shape)
    print(Ytr.shape)
    print(Xte.shape)
    print(Yte.shape)
