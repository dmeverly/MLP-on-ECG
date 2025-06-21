from .Layer import Layer
from .TanhLayer import TanhLayer
from .LogisticSigmoidLayer import LogisticSigmoidLayer
import numpy as np

e = 10**-7

class RecurrentNNLayer(Layer):

    #input: sizeIn = number of features coming in (codomain)
    #input: sizeOut = number of features coming out (domain)
    #output: none
    def __init__(self, sizeIn, sizeOut, eta):
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        eInputHidden = np.sqrt(6.0/(sizeIn+sizeOut))
        eHiddenHidden = np.sqrt(6.0/(sizeOut + sizeOut))
        self.tanh = TanhLayer()
        self.sigmoid = LogisticSigmoidLayer()
        self.eta = eta
        self.prevHidden = None
        self.wh = np.random.uniform(-eHiddenHidden,eHiddenHidden, [self.sizeOut, self.sizeOut])
        self.setWeights(np.random.uniform(-eInputHidden,eInputHidden, [sizeIn, self.sizeOut]))
        self.setBiases(np.zeros((1, sizeOut)))

        self.inputs = []
        self.hiddenStates = []
    
    def resetStates(self, n):
        self.inputs = []
        self.hiddenStates = []
        self.h0 = np.zeros((n, self.sizeOut))
    
    #input: none
    #output: sizeIn x sizeOut weight matrix
    def getHiddenWeights(self):
        return self.wh
    
    #input: sizeIn x sizeOut weight matrix   
    #output: none 
    def setHiddenWeights(self, weights):
        self.wh = weights

    def getPrevHidden(self):
        return self.prevHidden
    
    def setPrevHidden(self, state):
        self.prevHidden = state
    
    #input: none
    #output: sizeIn x sizeOut weight matrix
    def getWeights(self):
        return self.w
    
    #input: sizeIn x sizeOut weight matrix   
    #output: none 
    def setWeights(self, weights):
        self.w = weights

    #input: none
    #output: 1 x sizeOut bias vector    
    def getBiases(self):
        return self.b

    #input: 1 x sizeOut bias vector
    #output: none    
    def setBiases(self, biases):
        self.b = biases

    #input: dataIn, 1 x D data matrix
    #output: 1 x K data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N,T,_ = dataIn.shape
        self.prevHidden = np.zeros((N, self.sizeOut))
        ht = self.prevHidden
        self.inputs.clear()
        self.hiddenStates.clear()

        for t in range(T):
            xt = dataIn[:,t,:]
            z = xt@self.getWeights()+ht@self.getHiddenWeights()+self.getBiases()
            ht = self.tanh.forward(z)
            self.inputs.append(xt)
            self.hiddenStates.append(ht)
        
        self.prevHidden = ht
        self.setPrevOut(ht)
        return ht

    #input: none
    #output: an N x K x D tensor
    def gradient(self):
        h = self.getPrevOut()
        N = h.shape[0]
        #D, K = self.getWeights().shape
        #grad = np.zeros((N,K,D))
        W = self.getWeights().T
        print(W)
        return np.tile(W,(N,1,1))
    
    def backward2(self, djdh):
        T = len(self.inputs)
        N = djdh.shape[0]
        djdw = np.zeros_like(self.getWeights())
        djdwh = np.zeros_like(self.getHiddenWeights())
        djdb = np.zeros_like(self.getBiases())
        
        djdx = np.zeros((N,T,self.sizeIn))
        dh_next = djdh
        
        for t in reversed(range(T)):
            ht = self.hiddenStates[t]
            htprev = self.hiddenStates[t-1] if t > 0 else np.zeros_like(ht)
            xt = self.inputs[t]

            dz = dh_next * (1 - ht**2)

            djdw += xt.T @ dz
            djdwh += htprev.T @ dz
            djdb += np.sum(dz, axis=0, keepdims=True)

            djdx[:, t, :] = dz @ self.getWeights().T
            dh_next = dz @ self.getHiddenWeights().T

        self.setWeights(self.getWeights() - self.eta * djdw/N)
        self.setHiddenWeights(self.getHiddenWeights() - self.eta * djdwh / N)
        self.setBiases(self.getBiases() - self.eta * djdb / N)

        return djdx