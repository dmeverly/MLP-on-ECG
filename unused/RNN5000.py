# from framework import (
#     InputLayer,
#     FullyConnectedLayer,
#     LogisticSigmoidLayer,
#     LogLoss,
#     TanhLayer,
#     RecurrentNNLayer,
# )

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as pp
# import time

# TESTDATAPATH = 'Data/ECG5000_TEST.txt'
# TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'
# EPOCH_LIMIT = 10000
# ETA = 10**(-4)
# e = 10**(-7)
# hiddenStateSize = 8
# delimeter = 0.7

# def readData(path):
#     return pd.read_csv(path, sep=r'\s+', header=None)

# def preProcess(df):
#     df = df.sample(frac=1).reset_index(drop=True)
#     x = df.iloc[:,:]
#     x = df.drop([0], axis=1)
#     y = df.iloc[:,0]
#     y = y.where(y == 1, 0)
#     return x, y

# def splitData(X,Y):
#     n = X.shape[0]
#     split = int(np.floor(n/3))
#     validX = X.iloc[:split]
#     trainX = X.iloc[split:]
#     validY = np.atleast_2d(Y.iloc[:split]).T
#     trainY = np.atleast_2d(Y.iloc[split:]).T
#     trainX = trainX.to_numpy()
#     validX = validX.to_numpy()
#     return validX, trainX, validY, trainY

# def createFigure(epochs, trainEval, validEval):
#     figure, axis = pp.subplots()
#     axis.plot(epochs, trainEval, label='Training MSE')
#     axis.plot(epochs, validEval, label="Validation MSE")
#     pp.xlabel("Epochs")
#     pp.ylabel("MSE")
#     pp.legend()
#     pp.show()

# def forward(x, Layers):
#     h = x
#     for layer in Layers:
#         h = layer.forward(h)
#     return h

# def backward(grad, Layers):
#     for i in range(len(Layers)-2,0,-1):
#         newgrad = Layers[i].backward(grad)
            
#         if(isinstance(Layers[i],FullyConnectedLayer)):
#             Layers[i].updateWeights(grad,ETA)
            
#         grad = newgrad
#     return grad

# def backward2(grad, Layers):
#     for i in range(len(Layers)-2,0,-1):
#         newgrad = Layers[i].backward2(grad)
            
#         if isinstance(Layers[i],FullyConnectedLayer):
#             Layers[i].updateWeights(grad,ETA)
            
#         grad = newgrad
#     return grad

# def calculateStats(Y, Yhat):
#     rmse = np.sqrt(np.mean((Y-Yhat)**2))
#     smape = np.mean(2*np.abs(Yhat-Y)/(np.abs(Yhat)+np.abs(Y)+e))
#     Yhat = np.where(Yhat >= delimeter, 1, 0)
#     tp = np.sum((Yhat == 1) & (Y == 1))
#     fp = np.sum((Yhat == 1) & (Y == 0))
#     tn = np.sum((Yhat == 0) & (Y == 0))
#     fn = np.sum((Yhat == 0) & (Y == 1))

#     sensitivity = tp/(tp+fn) if (tp+fn > 0) else 0.0
#     specificity = tn/(tn+fp) if (tn+fp > 0) else 0.0

#     return rmse, smape, sensitivity, specificity

# def createConfusion(validY, predY, title):
#     predY = np.where(predY >= delimeter, 1, 0)
#     predY = np.ravel(predY)
#     validY = np.ravel(validY)
#     classes = np.unique(validY)
#     rows = len(classes)
#     confMatrix = np.zeros((rows,rows), dtype=int)
#     for i in range(len(predY)):
#         vi = validY[i]
#         vi = int(vi)
#         pi = predY[i]
#         pi = int(pi)
#         confMatrix[vi,pi] += 1
       
#     pp.figure(figsize=(6,6))
#     pp.imshow(confMatrix, interpolation='nearest', cmap=pp.cm.Blues)
#     pp.title(title)
#     pp.colorbar()
#     tick_marks = np.arange(len(classes))
#     pp.xticks(tick_marks, classes)
#     pp.yticks(tick_marks, classes)
#     pp.xlabel('Predicted Label')
#     pp.ylabel('True Label')

    
#     thresh = confMatrix.max() / 2
#     for i in range(len(classes)):
#         for j in range(len(classes)):
#             pp.text(j, i, format(confMatrix[i, j], 'd'),
#                     ha="center", va="center",
#                     color="white" if confMatrix[i, j] > thresh else "black")

#     pp.show()

# def zScore(y):
#     mean = np.atleast_2d(np.mean(y, axis = 0))
#     std = np.atleast_2d(np.std(y, axis = 0, ddof = 1))
#     return (y-mean)/(std+e), mean, std

# def zScoreValid(y, mean, std):
#     return (np.atleast_2d(y)-mean)/(std+e)

# def reverseZScore(y, mean, std):
#     return np.atleast_2d(y*std+mean)

# def createSaliency(x,y):
#     sampleYhat = forward(x, Layers)
#     sampleGrad = Layers[-1].gradient(y, sampleYhat)
#     print(sampleGrad)
#     exit(0)
#     _=backward2(sampleGrad, Layers)

#     saliency = np.abs(Layers[0].getGrad()).reshape(-1)
#     tracing = x.reshape(-1)
#     tracing = reverseZScore(tracing, L1.meanX.reshape(-1), L1.stdX.reshape(-1))

#     saliency = saliency.astype(np.float32)/np.max(saliency)

#     pp.figure(figsize=(10, 4))
#     pp.plot(tracing, label="ECG")
#     pp.plot(saliency, label="Saliency", alpha=0.6)
#     pp.fill_between(range(len(saliency)), 0, saliency, color="red", alpha=0.3)
#     pp.legend()
#     pp.title("Saliency Map on ECG")
#     pp.xlabel("Time Step")
#     pp.ylabel("Signal / Importance")
#     pp.show()

# if __name__ == "__main__":
#     np.random.seed(0)
    
#     #original dataset split valid/train 1/2:1/2 ratio
#     originTest = readData(TESTDATAPATH)
#     originTrain = readData(TRAINDATAPATH)
#     originData = pd.concat([originTest, originTrain], axis=0)
#     X, Y = preProcess(originData)
#     validX, trainX, validY, trainY = splitData(X,Y)
#     validX = np.expand_dims(validX.to_numpy(), axis =2)
#     trainX = np.expand_dims(trainX.to_numpy(), axis=2)

#     L1 = InputLayer(trainX)
#     L2 = RecurrentNNLayer(trainX.shape[2], hiddenStateSize, ETA)
#     L3 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize)
#     L4 = TanhLayer()
#     L5 = FullyConnectedLayer(hiddenStateSize, trainY.shape[1])
#     L6 = LogisticSigmoidLayer()
#     L7 = LogLoss()

#     Layers = [L1, L2, L3, L4, L5, L6, L7]
#     sqErrorTrain = []
#     sqErrorValid = []
#     epochs = []
#     epoch = 0
#     delta = 0

#     tic = time.perf_counter()
#     while(epoch < EPOCH_LIMIT):
#         epoch += 1

#         #training data forward-backward pass, training the weightns ad biases of FCC
#         trainYhat = forward(trainX, Layers)
#         j = Layers[-1].eval(trainY,trainYhat)
#         sqErrorTrain.append(j)
#         grad = Layers[-1].gradient(trainY, trainYhat)
#         backward2(grad, Layers)

#         #validation data forward pass
#         validYhat = forward(validX, Layers)
#         j = Layers[-1].eval(validY,validYhat)
#         sqErrorValid.append(j)
        
#         #runs at least 1000 epochs, if MSE increases 10 consecutive times, stop
#         epochs.append(epoch)
#         if epoch > 1000:
#             change = sqErrorTrain[-1]-sqErrorTrain[-2]
#             #break if change is less than 10^-7
#             if  np.abs(change) < 10**(-7):
#                 break
#             if sqErrorTrain[-1] > sqErrorTrain[-2]:
#                 delta += 1
#                 if delta >= 10:
#                     print("MSE increasing on Training Data")
#                     break
#             if sqErrorValid[-1] > sqErrorValid[-2]:
#                 delta += 1
#                 if delta >= 10:
#                     print("MSE increasing on Validation Data")
#                     break
#             else:
#                 delta = 0
#         if epoch % 10000 == 0:
#             toc = time.perf_counter()
#             print(f"Training MSE at {epoch} epochs was {sqErrorTrain[-1]}\n{np.round((toc - tic),2)} seconds have elapsed")
#             print(f"Validation MSE at {epoch} epochs was {sqErrorValid[-1]}\n{np.round((toc - tic),2)} seconds have elapsed")
#     toc = time.perf_counter()
#     print(f"training took {toc-tic} seconds")
#     trainRMSE, trainSMAPE, trainSensitivity, trainSpecificity = calculateStats(trainY, trainYhat)
#     validRMSE, validSMAPE, validSensitivity, validSpecificity = calculateStats(validY, validYhat)
#     #createSaliency(trainX[0:1], trainY[0:1])
#     print(f"Training RMSE = {trainRMSE} \nTraining SMAPE = {trainSMAPE} \nTraining Sensitivity = {trainSensitivity} \nTraining Specificity = {trainSpecificity}")
#     print(f"Validat. RMSE = {validRMSE} \nValidat. SMAPE = {validSMAPE} \nValidat. Sensitivity = {validSensitivity} \nValidat. Specificity = {validSpecificity}")
#     createFigure(epochs, sqErrorTrain, sqErrorValid)
#     createConfusion(trainY, trainYhat, "Training Confusion Matrix")
#     createConfusion(validY, validYhat, "Validation Confusion Matrix")
#     exit(0)