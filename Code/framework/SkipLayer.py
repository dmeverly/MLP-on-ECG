from .Layer import Layer
from .FullyConnectedLayer import FullyConnectedLayer
import numpy as np

class SkipLayer(Layer):
    #input: none
    #output: none
    def __init__(self, layerType):
        super().__init__()
        self.layerType = layerType
        self.projectionLayer = None

    #input: dataIn, a 1 x k matrix
    #output: a 1 x k matrix
    def forward(self, dataIn):
        h = dataIn
        self.setPrevIn(h)
        h = self.layerType.forward(h)
        if h.shape[1] != dataIn.shape[1]:
            if self.projectionLayer == None:
                self.projectionLayer = FullyConnectedLayer(dataIn.shape[1], h.shape[1])
            dataIn = self.projectionLayer.forward(dataIn)

        self.setPrevOut(h + dataIn)
        return self.getPrevOut()

    def backward(self, gradIn):     
        self.grad = self.layerType.backward(gradIn)
        if self.projectionLayer is not None:
                self.skipGrad = self.projectionLayer.backward(gradIn)
        else:
            self.skipGrad = gradIn
        return self.grad + self.skipGrad

    def backward2(self, gradIn):     
        return self.backward(gradIn)
    
    def gradient(self):
        return super().gradient()
    
    def updateWeights(self, grad, eta):
        if hasattr(self.layerType, 'updateWeights'):
            self.layerType.updateWeights(self.grad, eta)
        if self.projectionLayer is not None:
            self.projectionLayer.updateWeights(self.skipGrad, eta)