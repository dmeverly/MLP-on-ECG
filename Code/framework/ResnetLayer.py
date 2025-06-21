from .Layer import Layer
from framework import(
    FullyConnectedLayer,
    ReLULayer,
    BatchNormLayer
) 
import numpy as np

class ResnetLayer(Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = FullyConnectedLayer(dim, dim)
        self.bn1 = BatchNormLayer(dim)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(dim, dim)
        self.bn2 = BatchNormLayer(dim)
        self.projection = None

    def forward(self, x):
        self.setPrevIn(x)
        out = self.fc1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        out = self.bn2.forward(out)

        if out.shape[1] != x.shape[1]:
            self.projection = self.projection or FullyConnectedLayer(x.shape[1], out.shape[1])
            x = self.projection.forward(x)

        res = self.relu.forward(out + x)
        self.setPrevOut(res)
        return res

    def backward(self, grad):
        grad_res = self.relu.backward(grad)
        grad_out = self.bn2.backward(grad_res)
        grad_out = self.fc2.backward(grad_out)
        grad_out = self.relu.backward(self.bn1.backward(self.fc1.backward(grad_out)))

        grad_skip = self.projection.backward(grad_res) if self.projection else grad_res
        return grad_out + grad_skip

    def updateWeights(self, _, eta):
        for layer in [self.fc1, self.bn1, self.fc2, self.bn2]:
            if hasattr(layer, 'updateWeights'):
                layer.updateWeights(layer.grad, eta)
        if self.projection:
            self.projection.updateWeights(self.grad, eta)