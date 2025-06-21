from .Layer import Layer
import numpy as np

class LinearLayer(Layer):
    #input: none
    #output: none
    def __init__(self):
        super().__init__()

    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        h = dataIn
        self.setPrevOut(h)
        return h

    def gradient(self):
        pass