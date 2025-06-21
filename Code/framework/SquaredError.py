import numpy as np

e = 10**(-7)

class SquaredError():
    def __init__(self):
        pass
    #input: Y is an N x K matrix of target values
    #input: Yhat is an N x K matrix of estimated values
    #where N is any integer >= 1
    #Output: a single floating point value
    def eval(self,Y,Yhat):
        return np.mean((Y-Yhat)**2)

    #input: Y is an N x K matrix of target values
    #input: Yhat is an N x L matrix of estimated values
    #output: N x K matrix
    def gradient(self, Y, Yhat):
        return np.atleast_2d(-2*(Y-Yhat))
    
    #returns input unchanged
    def forward(self, Yhat):
        return Yhat