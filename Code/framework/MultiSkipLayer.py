from .Layer import Layer
from framework import (
    FullyConnectedLayer,
    SkipLayer
)

class MultiSkipLayer(SkipLayer):
    def __init__(self, layerType):
        layers = layerType if isinstance(layerType, (list, tuple)) else [layerType]
        super().__init__(layers)
        self.layers = layers
        self.projectionLayer = None

    def forward(self, x):
        self.setPrevIn(x)
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        if out.shape[1] != x.shape[1]:
            if self.projectionLayer is None:
                self.projectionLayer = FullyConnectedLayer(x.shape[1], out.shape[1])
            x = self.projectionLayer.forward(x)
        res = out + x
        self.setPrevOut(res)
        return res

    def backward(self, grad):
        grad_main = grad
        for layer in reversed(self.layers):
            grad_main = layer.backward(grad_main)
        grad_skip = self.projectionLayer.backward(grad) if self.projectionLayer else grad
        return grad_main + grad_skip

    def updateWeights(self, grad, eta):
        for layer in self.layers:
            if hasattr(layer, 'updateWeights'):
                layer.updateWeights(layer.grad, eta)
        if self.projectionLayer is not None:
            self.projectionLayer.updateWeights(grad, eta)