from .Layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):

    #input: sizeIn = number of features coming in (codomain)
    #input: sizeOut = number of features coming out (domain)
    #output: none
    def __init__(self, sizeIn, sizeOut, ADAM = False):
        super().__init__()
        e = np.sqrt(6/(sizeIn+sizeOut))
        self.setWeights(np.random.uniform(-e,e, [sizeIn, sizeOut]))
        self.setBiases(np.random.uniform(-e,e, [1, sizeOut]))
        self.ADAM = ADAM

        if(ADAM):
            self.callCount = 0
            self.p1 = 0.9
            self.p2 = 0.999
            self.delta = 10**-8
            self.sw = np.zeros((sizeIn,sizeOut))
            self.rw = np.zeros((sizeIn,sizeOut))
            self.sb = np.zeros((1,sizeOut))
            self.rb = np.zeros((1,sizeOut))
    
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
        h = np.dot(dataIn,self.getWeights())+self.getBiases()
        self.setPrevOut(h)
        return h

    #input: none
    #output: an N x K x D tensor
    def gradient(self):
        h = self.getPrevOut()
        N = h.shape[0]
        W = self.getWeights().T
        return np.tile(W,(N,1,1))
    
    def updateWeights(self, djdh, eta):
        N = djdh.shape[0]
        djdw = (self.getPrevIn().T@djdh)/N
        djdb = np.mean(djdh, axis=0)

        if(self.ADAM):
            self.callCount += 1

            #for weights
            self.sw = self.p1*self.sw+(1-self.p1)*djdw
            swh = self.sw/(1-self.p1**(self.callCount))
            self.rw = self.p2*self.rw+(1-self.p2)*(djdw*djdw)
            rwh = self.rw/(1-self.p2**(self.callCount))
            adam = swh / (np.sqrt(rwh)+self.delta)
            self.setWeights(self.getWeights()-eta*adam)

            #for biases
            self.sb = self.p1*self.sb+(1-self.p1)*djdb
            sbh = self.sb/(1-self.p1**(self.callCount))
            self.rb = self.p2*self.rb+(1-self.p2)*(djdb*djdb)
            rbh = self.rb/(1-self.p2**(self.callCount))
            adam = sbh / (np.sqrt(rbh)+self.delta)
            self.setBiases(self.getBiases()-eta*adam)

        else:
            self.setWeights(self.getWeights() - eta*djdw)
            self.setBiases(self.getBiases() - eta*djdb)
