from .Layer import Layer
import numpy as np

class ReLULayer(Layer):
    #input: none
    #output: none
    def __init__(self):
        super().__init__()

    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        h = dataIn
        self.setPrevIn(h)
        h = np.where(h <= 0, 0, h)
        self.setPrevOut(h)
        return h

    def gradient(self):
        h = self.getPrevIn()
        N,K = h.shape
        grad = np.zeros((N,K,K))

        for i in range(N):
            grad[i] = np.diagflat((h[i]>0).astype(float))
        
        return grad