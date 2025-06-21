import numpy as np
from .Layer import Layer

class BatchNormLayer(Layer):
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__()
        self.gamma = np.ones((1, dim))
        self.beta  = np.zeros((1, dim))
        self.eps = eps
        self.momentum = momentum
        self.mean = np.zeros((1, dim))
        self.var  = np.ones((1, dim))

    def forward(self, x, train=True):
        if train:
            mean = x.mean(axis=0, keepdims=True)
            var  = x.var(axis=0, keepdims=True)
            self.mean = (
                self.momentum * self.mean
                + (1 - self.momentum) * mean
            )
            self.var = (
                self.momentum * self.var
                + (1 - self.momentum) * var
            )
            mu, var = mean, var
        else:
            mu, var = self.mean, self.var

        self.x_centered = x - mu
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        x_norm = self.x_centered * self.std_inv
        out = self.gamma * x_norm + self.beta

        self.setPrevIn(x)
        self.setPrevOut(out)
        return out

    def backward(self, dout):
        N, D = dout.shape
        self.gamma_grad = np.sum(dout * (self.x_centered * self.std_inv), axis=0, keepdims=True)
        self.beta_grad  = np.sum(dout, axis=0, keepdims=True)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * self.x_centered, axis=0, keepdims=True) * -0.5 * (self.std_inv**3)
        dmean = (
            np.sum(dx_norm * -self.std_inv, axis=0, keepdims=True)
            + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)
        )
        dx = (
            dx_norm * self.std_inv
            + dvar * 2 * self.x_centered / N
            + dmean / N
        )
        return dx

    def updateWeights(self, grad, eta):
        self.gamma -= eta * self.gamma_grad
        self.beta  -= eta * self.beta_grad

    def gradient(self):
        pass