import numpy as np
from .Layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()
    
    #Input: N x H x W tensor
    #Output: flattened matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N,H,W = dataIn.shape
        out = dataIn.reshape(N, H*W, order ='F')

        self.setPrevOut(out)
        return out
    
    def gradient(self, gradIn):
        gradOut = gradIn.reshape(self.getPrevIn().shape, order='F')
        return gradOut

    def backward(self, gradIn):
        return self.gradient(gradIn)