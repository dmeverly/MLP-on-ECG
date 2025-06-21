from .Layer import Layer
import numpy as np

class MaxPoolLayer(Layer):
    def __init__(self, poolSize, stride):
        super().__init__()
        self.poolSize = poolSize
        self.stride = stride
    
    #Input: N x H x W tensor
    #Output: N x H x W tensor representing max-pooling
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N,H,W = dataIn.shape
        M = self.poolSize
        S = self.stride
        Oh = (H - M)//S + 1
        Ow = (W - M)//S + 1
        out = np.empty((N, Oh, Ow))

        self.maxValIdx = np.empty((N,Oh,Ow,2),dtype=np.int32)

        for n in range(N):
            for i in range(Oh):
                for j in range(Ow):
                    boxStartx = i * S
                    boxEndx = boxStartx + M
                    boxStarty = j * S
                    boxEndy = boxStarty + M
                    cells = dataIn[n, boxStartx:boxEndx, boxStarty:boxEndy]
                    maxIdx = np.argmax(cells)
                    maxVal = cells.flat[maxIdx]
                    maxIDxPos = np.unravel_index(maxIdx, (M,M))
                    yIDx = boxStartx + maxIDxPos[0]
                    xIDx = boxStarty + maxIDxPos[1]
                    self.maxValIdx[n,i,j]= [yIDx,xIDx]
                    out[n,i,j] = maxVal
        self.setPrevOut(out)
        return out
    
    def gradient(self, gradIn):
        N, Oh, Ow = gradIn.shape
        gradOut = np.zeros_like(self.getPrevIn())

        for n in range(N):
            for i in range(Oh):
                for j in range(Ow):
                    x,y = self.maxValIdx[n,i,j]
                    gradOut[n,x,y] = gradIn[n,i,j]
        return gradOut

    def backward(self, gradIn):
        return self.gradient(gradIn)