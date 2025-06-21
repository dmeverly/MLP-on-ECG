import numpy as np

e = 10**(-7)

class LogLoss():
    def __init__(self):
        pass
    #input: Y is an N x K matrix of target values
    #input: Yhat is an N x K matrix of estimated values
    #where N is any integer >= 1
    #Output: a single floating point value
    def eval(self,Y,Yhat):
        j = -((Y*np.log(Yhat+e))+((1-Y)*np.log(1-Yhat+e)))
        mean = np.mean(j)
        return mean

    #input: Y is an N x K matrix of target values
    #input: Yhat is an N x L matrix of estimated values
    #output: N x K matrix
    def gradient(self, Y, Yhat):
        grad = (Yhat-Y)/((Yhat*(1-Yhat))+e)
        return grad
    
    #returns input unchanged
    def forward(self, Yhat):
        return Yhat