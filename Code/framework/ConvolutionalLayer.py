import numpy as np
from .Layer import Layer
import time
from numba import njit

eps = 10**-2

# try threading for speed!
@njit
def threadGradient(gradIn, prevIn, invKernel, M):
    N, K, D = prevIn().shape
    djdx = np.empty(N,K,D)
    for n in range(N):
        padded = np.pad(gradIn[n], ((M-1, M-1), (M-1,M-1)), mode='constant')
        for i in range(K):
            for j in range(D):
                djdx[n,i,j] = np.sum(padded[i:i+M,j:j+M]*invKernel)
    return djdx

@njit
def threadKernelGradient(gradIn, prevIn, kernels, M):
    N, Oh, Ow = gradIn.shape
    djdf = np.empty_like(kernels)

    for n in range(N):
        for i in range(Oh):
            for j in range(Ow):
                djdf += gradIn[n,i,j] * prevIn[n,i:i+M, j:j+M]
    
    return djdf

    
class ConvolutionalLayer(Layer):
    def __init__(self, kernelSize):
        super().__init__()
        self.kernelSize = kernelSize
        self.setKernels(np.random.uniform(-eps,eps, [kernelSize,kernelSize]))
    
    def getKernels(self):
        return self.kernels
    
    #take a matrix or tensor and sets kernels
    def setKernels(self, weights):
        w = weights
        self.kernels = w
    
    @staticmethod
    def crossCorrelate2D(kernel, matrixA):
        H, W = matrixA.shape
        M, _ = kernel.shape
        Oh = H - M + 1
        Ow = W - M + 1
        CC = np.zeros((Oh, Ow))

        #F = sum i,j to m X(a+i-1,b+j-1)K(i,j)
        for i in range(Oh):
            for j in range(Ow):
                cells = matrixA[i:i+M, j:j+M].flatten()
                CC[i,j] = np.dot(cells, kernel.flatten())
        return CC
    
    #Input: N x H x W tensor
    #Output: N x H x W tensor representing cross correlation
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N,H,W = dataIn.shape
        M,_ = self.getKernels().shape
        Oh = H - M + 1
        Ow = W - M + 1
        out = np.zeros((N, Oh, Ow))
        for i in range(N):
            out[i] = ConvolutionalLayer.crossCorrelate2D(self.getKernels(), dataIn[i])
        self.setPrevOut(out)
        return out
    '''
    def gradient(self, gradIn):
        #start = time.time()
        invKernel = np.rot90(self.kernels,2)
        M = self.kernelSize
        djdx = np.zeros_like(self.getPrevIn())

        

        for n in range(N):
            padded = np.pad(gradIn[n], ((M-1, M-1), (M-1,M-1)), mode='constant')
            for i in range(K):
                for j in range(D):
                    djdx[n,i,j] = np.sum(padded[i:i+M,j:j+M]*invKernel)
        #stop = time.time()
        #print(f"grad took {stop-start} seconds")
        return djdx
    '''
    #threaded version
    def gradient(self, gradIn):
        #start = time.time()
        invKernel = np.rot90(self.kernels,2)
        M = self.kernelSize
        djdx = np.zeros_like(self.getPrevIn())

        N, K, D = self.getPrevIn().shape

        for n in range(N):
            padded = np.pad(gradIn[n], ((M-1, M-1), (M-1,M-1)), mode='constant')
            for i in range(K):
                for j in range(D):
                    djdx[n,i,j] = np.sum(padded[i:i+M,j:j+M]*invKernel)
        #stop = time.time()
        #print(f"grad took {stop-start} seconds")
        return djdx
    
    def backward(self, gradIn):
        return self.gradient(gradIn)
    '''
    def updateKernels(self, gradIn, eta):
        if eta <= 0:
            return
        
        #start = time.time()
        N, Oh, Ow = gradIn.shape
        X = self.getPrevIn()
        M = self.kernelSize
        self.djdf = np.zeros_like(self.kernels)

        for n in range(N):
            for i in range(Oh):
                for j in range(Ow):
                    self.djdf += gradIn[n,i,j] * X[n,i:i+M, j:j+M]

        self.setKernels(self.getKernels()-eta*self.djdf)
        #stop = time.time()
        #print(f"update took {stop-start} seconds")
    '''

    #threaded version
    def updateKernels(self, gradIn, eta):
        if eta <= 0:
            return
        
        M = self.kernelSize
        self.djdf = threadKernelGradient(gradIn,self.getPrevIn(),self.kernels,M)

        self.setKernels(self.getKernels()-eta*self.djdf)