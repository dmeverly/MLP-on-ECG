import numpy as np
from .Layer import Layer

e = 10**(-7)

class InputLayer(Layer):
    #input: dataIn, an (N by D) matrix
    #output: none
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.atleast_2d(np.mean(dataIn, axis= 0))
        self.stdX = np.atleast_2d(np.std(dataIn, axis = 0, ddof=1))
        self.stdX = np.where(self.stdX == 0, 1, self.stdX)

    #Input : dataIn, A (1 by D) matrix
    #Output : A (1 by D) matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        h = (dataIn-self.meanX)/(self.stdX+e)
        self.setPrevOut(h)
        return h

    def gradient(self):
        pass