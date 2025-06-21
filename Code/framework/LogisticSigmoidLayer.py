from .Layer import Layer
import numpy as np

e = 10**(-7)

class LogisticSigmoidLayer(Layer):
    #input: none
    #output: none
    def __init__(self):
        super().__init__()
    
    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        h = 1/(1+np.exp(-dataIn)+e)
        h = np.clip(h, e, 1.0-e)
        h = np.nan_to_num(h, nan=e, posinf=1.0-e, neginf=e)
        self.setPrevOut(h)
        return h

    #input: none
    #output: an N x K x D tensor
    '''
    def gradient(self):
        h = self.getPrevOut()  
        grad_diag = h * (1 - h) 
        grad = np.zeros((h.shape[0], h.shape[1], h.shape[1]))
        idx = np.arange(h.shape[1])
        grad[:, idx, idx] = grad_diag
        return grad
    '''
    #was gradient2
    def gradient(self):
        h = self.getPrevOut()
        return h*(1-h)
    
    #was backward2
    def backward(self, djdz):
        grad = self.gradient()
        return djdz * grad