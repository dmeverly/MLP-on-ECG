from .Layer import Layer
import numpy as np

e = 10**(-7)

class TanhLayer(Layer):
    #input: none
    #output: none
    def __init__(self):
        super().__init__()

    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        h = (np.exp(dataIn)-np.exp(-dataIn))/(np.exp(dataIn)+np.exp(-dataIn)+e)
        self.setPrevOut(h)
        return h

    def gradient(self):
        h = self.getPrevOut()
        grad_diag = 1 - h**2  
        grad = np.zeros((h.shape[0], h.shape[1], h.shape[1]))  
        idx = np.arange(h.shape[1])
        grad[:, idx, idx] = grad_diag
        return grad
    
    def gradient2(self):
        h = self.getPrevOut()
        grad = 1-h**2
        return grad
    
    def backward(self, djdz):
        h = self.getPrevOut()
        return djdz * (1 - h ** 2)

