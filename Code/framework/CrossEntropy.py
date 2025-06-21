import numpy as np

e = 10**(-7)

class CrossEntropy():
    def __init__(self):
        pass

    def eval(self,Y,Yhat):
        j = -np.sum(Y*np.log(Yhat+e))/Y.shape[0]
        return j

    def gradient(self, Y, Yhat):
        grad = -(Y/(Yhat+e))
        return grad
    
    #returns input unchanged
    def forward(self, Yhat):
        return Yhat