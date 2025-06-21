from .Layer import Layer
import numpy as np

e = 10**(-7)

class SoftmaxLayer(Layer):
    #input: none
    #output: none
    def __init__(self):
        super().__init__()

    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        temp = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exp = np.exp(temp)
        h = exp / (np.sum(exp, axis=1, keepdims=True)+e)
        self.setPrevOut(h)
        return h

    def gradient(self):
        x = self.getPrevOut()
        N, k = x.shape
        dgdz = np.zeros((N,k,k))

        for n in range(N):
            h = x[n].reshape(-1,1)
            dgdz[n] = np.diagflat(h) - h@h.T
        
        return dgdz