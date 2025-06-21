from framework import (
    InputLayer,
    FullyConnectedLayer,
    LogisticSigmoidLayer,
    LogLoss,
    TanhLayer,
    BatchNormLayer,
    MultiSkipLayer,
    ReLULayer
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import time
import wfdb
import os
import random
import sys
import argparse
import re

EPOCH_LIMIT = 20000
hiddenStateSizes = [32]
ETAs = [10**-5]
e = 10**(-7)

def parseArguments():
    parser = argparse.ArgumentParser(description="ECG Project")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parserECG200 = subparsers.add_parser("ecg200", help="Run ecg 200")
    parserECG5000 = subparsers.add_parser("ecg5000", help="Run ecg 5000")
    parserECG5000Multiclass = subparsers.add_parser('ecg5000multiclass', help="Run ecg 5000 multiclass")
    parserMIT = subparsers.add_parser("mit", help="Run MIT")
    parserFull = subparsers.add_parser("full", help="Run full ECG")

    return parser.parse_args()

def readData(path):
    return pd.read_csv(path, sep=r'\s+', header=None)

def preProcess(df, ohc = False):
    df = df.sample(frac=1).reset_index(drop=True)
    x = df.iloc[:,:]
    x = df.drop([0], axis=1)
    y = df.iloc[:,0]
    if ohc:
        y = OneHotEncoding(y)
    else:
        y = y.where(y == 1, 0)
    return x, y

def splitData(X,Y):
    n = X.shape[0]
    split = int(np.floor(n/3))
    validX = X.iloc[:split]
    trainX = X.iloc[split:]
    validY = np.atleast_2d(Y.iloc[:split]).T
    trainY = np.atleast_2d(Y.iloc[split:]).T
    trainX = trainX.to_numpy()
    validX = validX.to_numpy()
    return validX, trainX, validY, trainY

def splitDataNP(X,Y):
    n = X.shape[0]
    split = int(np.floor(n/3))
    validX = X[:split]
    trainX = X[split:]
    validY = np.atleast_2d(Y[:split])
    trainY = np.atleast_2d(Y[split:])
    
    return validX, trainX, validY, trainY

def createFigure(epochs, trainEval, validEval, title):
    figure, axis = pp.subplots()
    axis.plot(epochs, trainEval, label='Training MSE')
    axis.plot(epochs, validEval, label="Validation MSE")
    pp.xlabel("Epochs")
    pp.ylabel("Cross Entropy Loss")
    pp.title(title)
    pp.legend()

    outDir = os.path.join(os.path.dirname(os.getcwd()), 'Images')
    os.makedirs(outDir, exist_ok=True)
    fileName = re.sub(r'[^\w\-]+', '_', title.strip().lower()) + '.png'
    outPath = os.path.join(outDir, fileName)

    if outPath:
        figure.savefig(outPath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {outPath}")
    
    pp.close(figure)

def forward(x, Layers):
    h = x
    for layer in Layers:
        h = layer.forward(h)
    return h

def backward(grad, Layers, ETA):
    for i in range(len(Layers)-2,-1,-1):
        if(isinstance(Layers[i],InputLayer)):
            return grad
        newgrad = Layers[i].backward(grad)
            
        if(isinstance(Layers[i],FullyConnectedLayer)):
            Layers[i].updateWeights(grad,ETA)
            
        grad = newgrad
    return grad

def calculateStats(Y, Yhat, ):
    rmse = np.sqrt(np.mean((Y-Yhat)**2))
    smape = np.mean(2*np.abs(Yhat-Y)/(np.abs(Yhat)+np.abs(Y)+e))
    Yhat = np.where(Yhat >= 0.5, 1, 0)
    tp = np.sum((Yhat == 1) & (Y == 1))
    fp = np.sum((Yhat == 1) & (Y == 0))
    tn = np.sum((Yhat == 0) & (Y == 0))
    fn = np.sum((Yhat == 0) & (Y == 1))

    sensitivity = tp/(tp+fn) if (tp+fn > 0) else 0.0
    specificity = tn/(tn+fp) if (tn+fp > 0) else 0.0

    accuracy = np.mean(Y == Yhat)

    return rmse, smape, sensitivity, specificity, accuracy

def createConfusion(validY, predY, title, multiclass = False):
    prepend = "Confusion Matrix "
    title = prepend+title
    if not multiclass:
        predY = np.where(predY >= 0.5, 1, 0)
    else:
        predY = np.argmax(predY, axis=1)
        validY = np.argmax(validY, axis = 1)
    predY = np.ravel(predY)
    validY = np.ravel(validY)

    classes = np.unique(np.concatenate([validY]))
    classIndex = {label: i for i, label in enumerate(classes)}
    rows = len(classes)
    confMatrix = np.zeros((rows,rows), dtype=int)
    for i in range(len(predY)):
        vi = classIndex[validY[i]]
        pi = classIndex[predY[i]]
        confMatrix[vi,pi] += 1
       
    pp.figure(figsize=(6,6))
    pp.imshow(confMatrix, interpolation='nearest', cmap=pp.cm.Blues)
    pp.title(title)
    pp.colorbar()
    tick_marks = np.arange(len(classes))
    pp.xticks(tick_marks, classes)
    pp.yticks(tick_marks, classes)
    pp.xlabel('Predicted Label')
    pp.ylabel('True Label')

    
    thresh = confMatrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            pp.text(j, i, str(confMatrix[i, j]),
                    ha="center", va="center",
                    color="white" if confMatrix[i, j] > thresh else "black")
            
    outDir = os.path.join(os.path.dirname(os.getcwd()), 'Images')
    os.makedirs(outDir, exist_ok=True)
    fileName = re.sub(r'[^\w\-]+', '_', title.strip().lower()) + '_confusion.png'
    outPath = os.path.join(outDir, fileName)

    if outPath:
        figure = pp.gcf()
        figure.savefig(outPath, dpi=300, bbox_inches='tight')
        print(f"Confusion Matrix saved to {outPath}")
    
    pp.close(figure)


def zScore(y):
    mean = np.atleast_2d(np.mean(y, axis = 0))
    std = np.atleast_2d(np.std(y, axis = 0, ddof = 1))
    return (y-mean)/(std+e), mean, std

def zScoreValid(y, mean, std):
    return (np.atleast_2d(y)-mean)/(std+e)

def reverseZScore(y, mean, std):
    return np.atleast_2d(y*std+mean)

def train(validX, trainX, validY, trainY, Layers, title, ETA, multiclass = False):
    sqErrorTrain = []
    sqErrorValid = []
    epochs = []
    epoch = 0
    trainDelta = 0
    validDelta = 0
    bestLoss = np.inf
    delta = 0
    patience = 10

    tic = time.perf_counter()
    while(epoch < EPOCH_LIMIT):
        epoch += 1

        #training data forward-backward pass, training the weightns ad biases of FCC
        trainYhat = forward(trainX, Layers)
        j = Layers[-1].eval(trainY,trainYhat)
        sqErrorTrain.append(j)
        grad = Layers[-1].gradient(trainY, trainYhat)
        backward(grad, Layers, ETA)

        #validation data forward pass
        validYhat = forward(validX, Layers)
        jValid = Layers[-1].eval(validY,validYhat)
        sqErrorValid.append(jValid)
        
        #runs at least 1000 epochs, if MSE increases 10 consecutive times, stop
        epochs.append(epoch)
        if epoch > 1000:
            change = sqErrorTrain[-1]-sqErrorTrain[-2]
            #break if change is less than 10^-7
            if  np.abs(change) < 10**(-7):
                print(f"Stopping at {epoch} due to no learning on training data")
                break

            #break if error is increasing on training or validation
            if sqErrorTrain[-1] > sqErrorTrain[-2]:
                trainDelta += 1
                if trainDelta >= patience:
                    print(f"Stopping at {epoch} due to increasing error on training")
                    break
            else:
                trainDelta = 0
            if sqErrorValid[-1] > sqErrorValid[-2]:
                validDelta += 1
                if validDelta >= patience:
                    print(f"Stopping at {epoch} due to increasing error on validation")
                    break
            else:
                validDelta = 0
            
            #break if training and validation are diverging
            if j+jValid > bestLoss:
                delta +=1
                if delta >= patience:
                    print(f"Stopping at {epoch} due to divergence")
                    break
            else:
                bestLoss = j+jValid
                delta = 0

        if epoch % 10000 == 0:
            toc = time.perf_counter()
            print(f"{epoch} epochs ; {np.round((toc - tic),2)} seconds have elapsed")
            
    toc = time.perf_counter()
    print(f"training took {toc-tic} seconds")
    
    validAccuracy = printResults(trainY, trainYhat, validY, validYhat, title)
    printFigures(epochs, sqErrorTrain, sqErrorValid, trainY, trainYhat, validY, validYhat, multiclass, title)
    return validAccuracy

def printResults(trainY, trainYhat, validY, validYhat, title, ):
    trainRMSE, trainSMAPE, trainSensitivity, trainSpecificity, trainAccuracy = calculateStats(trainY, trainYhat, )
    validRMSE, validSMAPE, validSensitivity, validSpecificity, validAccuracy = calculateStats(validY, validYhat, )
    print(f"{title}\nTraining Sensitivity = {trainSensitivity*100:.2f}% \nTraining Specificity = {trainSpecificity*100:.2f}% \nTraining Accuracy = {trainAccuracy*100:.2f}%")
    print(f"Validat. Sensitivity = {validSensitivity*100:.2f}% \nValidat. Specificity = {validSpecificity*100:.2f}% \nValidat. Accuracy = {validAccuracy*100:.2f}%")
    return validAccuracy

def printFigures(epochs, sqErrorTrain, sqErrorValid, trainY, trainYhat, validY, validYhat, multiclass, title):
    createFigure(epochs, sqErrorTrain, sqErrorValid, title)
    createConfusion(trainY, trainYhat, title, multiclass)
    createConfusion(validY, validYhat, title, multiclass)

def OneHotEncoding(y):
    return pd.get_dummies(y)

def subsampleEqual(X, Y, random_state=None):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(Y, return_counts=True)

    n_minority = counts.min()
    indices = []

    for cls in classes:
        cls_indices = np.where(Y == cls)[0]
        cls_sample = rng.choice(cls_indices, size=n_minority, replace=False)
        indices.append(cls_sample)

    all_indices = np.concatenate(indices)
    rng.shuffle(all_indices)

    return X[all_indices], Y[all_indices]

def clip_extremes(df, n_std=5):
    means = df.mean()
    stds  = df.std(ddof=1)
    lower = means - n_std * stds
    upper = means + n_std * stds
    return df.clip(lower, upper, axis=1)

def clean_dataframe(df, drop_thresh=0.5, impute=False):
    col_keep = df.columns[df.isnull().mean() < drop_thresh]
    df = df[col_keep]

    if impute:
        df = df.fillna(df.mean())
    else:
        df = df.dropna()

    return df

def readFullECG(PATH1, PATH2, subsample_validation=False, random_state=None):
    # Load raw CSVs
    df_train = pd.read_csv(PATH1, header=0)
    df_valid = pd.read_csv(PATH2, header=0)

    # Clean them
    df_train = clean_dataframe(df_train, drop_thresh=0.5, impute=False)
    df_valid = clean_dataframe(df_valid, drop_thresh=0.5, impute=False)

    df_train = clip_extremes(df_train, n_std=5)
    df_valid = clip_extremes(df_valid, n_std=5)

    # Convert to NumPy
    arr_train = df_train.to_numpy()
    arr_valid = df_valid.to_numpy()

    # Split features/labels
    trainX = arr_train[:, 1:]
    trainY = arr_train[:, 0].astype(int)

    validX = arr_valid[:, 1:]
    validY = arr_valid[:, 0].astype(int)

    if subsample_validation:
        validX, validY = subsampleEqual(validX, validY, random_state)
        trainX, trainY = subsampleEqual(trainX, trainY, random_state)
    
    trainY = np.atleast_2d(trainY).T
    validY = np.atleast_2d(validY).T

    return validX, trainX, validY, trainY

def FullECGShallow(hiddenStateSize, ETA):

    PATH1 = 'Data/MI2023/TrainingDataset.csv'
    PATH2 = 'Data/MI2023/ValidationDataset.csv'
    
    validX, trainX, validY, trainY = readFullECG(PATH1, PATH2, subsample_validation=True)

    Layers = createFullPipelineShallow(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'FullECG Shallow with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def FullECG(hiddenStateSize, ETA):

    PATH1 = 'Data/MI2023/TrainingDataset.csv'
    PATH2 = 'Data/MI2023/ValidationDataset.csv'
    
    validX, trainX, validY, trainY = readFullECG(PATH1, PATH2, subsample_validation=True)

    Layers = createFullPipelineDeep(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'FullECG with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def FullECGSkip(hiddenStateSize, ETA):

    PATH1 = 'Data/MI2023/TrainingDataset.csv'
    PATH2 = 'Data/MI2023/ValidationDataset.csv'
    
    validX, trainX, validY, trainY = readFullECG(PATH1, PATH2, subsample_validation=True)

    Layers = createFullPipelineSkip(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'FullECG + Skip with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MLP200Shallow(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG200_TEST.txt'
    TRAINDATAPATH = 'Data/ECG200_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)
    validX, trainX, validY, trainY = splitData(X,Y)
    Layers = createPipelineShallow(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'MLP200 Shallow with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )
    
def MLP200(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG200_TEST.txt'
    TRAINDATAPATH = 'Data/ECG200_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)
    validX, trainX, validY, trainY = splitData(X,Y)
    Layers = createPipelineDeep(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'MLP200 with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MLP200Skip(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG200_TEST.txt'
    TRAINDATAPATH = 'Data/ECG200_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)
    validX, trainX, validY, trainY = splitData(X,Y)
    Layers = createPipelineSkip(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'MLP200 + Skip with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MLP5000Shallow(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'

    np.random.seed(0)

    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)
    validX, trainX, validY, trainY = splitData(X,Y)
    Layers = createPipelineShallow(trainX, trainY, hiddenStateSize)

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 Shallow with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )
    
def MLP5000(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'

    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)

    validX, trainX, validY, trainY = splitData(X,Y)

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    Layers = createPipelineDeep(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )



def MLP5000Skip(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData)
    validX, trainX, validY, trainY = splitData(X,Y)

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    Layers = createPipelineSkip(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 + Skip with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MLP5000MulticlassShallow(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData, ohc=True)
    validX, trainX, validY, trainY = splitData(X,Y)
    trainY = trainY.T
    validY = validY.T

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    Layers = createPipelineShallow(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 Multiclass Shallow with eta = {ETA} and hidden = {hiddenStateSize}', ETA, multiclass=True)

def MLP5000Multiclass(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData, ohc=True)
    validX, trainX, validY, trainY = splitData(X,Y)
    trainY = trainY.T
    validY = validY.T

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    Layers = createPipelineDeep(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 Multiclass with eta = {ETA} and hidden = {hiddenStateSize}', ETA,multiclass=True)


def MLP5000SkipMulticlass(hiddenStateSize, ETA, ):
    TESTDATAPATH = 'Data/ECG5000_TEST.txt'
    TRAINDATAPATH = 'Data/ECG5000_TRAIN.txt'
    
    np.random.seed(0)

    #original dataset split valid/train 1/2:1/2 ratio
    originTest = readData(TESTDATAPATH)
    originTrain = readData(TRAINDATAPATH)
    originData = pd.concat([originTest, originTrain], axis=0)
    X, Y = preProcess(originData, ohc=True)
    validX, trainX, validY, trainY = splitData(X,Y)
    trainY = trainY.T
    validY = validY.T

    validX, validY = subsampleEqual(validX, validY, 0)
    trainX, trainY = subsampleEqual(trainX, trainY, 0)

    Layers = createPipelineSkip(trainX, trainY, hiddenStateSize)
    return train(validX, trainX, validY, trainY, Layers, f'ss MLP5000 Multiclass + Skip with eta = {ETA} and hidden = {hiddenStateSize}', ETA,multiclass=True)

def readMitMatrix(folder, stripLength):

    #data recordings digitized at 250 Hz, stripLength * Hz = samples per segment
    hz = 250
    W = int(stripLength * hz)
    windows, labels = [], []

    # original symbols marking ST deviation in the .xws stream
    ischemia = {'s', 't', 'T'}

    for file in os.listdir(folder):
        if not file.endswith('.dat'):
            continue
        record = os.path.splitext(file)[0]
        recordPath = os.path.join(folder, record)
        record = wfdb.rdrecord(recordPath)
        
        #some data had only 1 lead, omit it
        if record.p_signal.shape[1] < 2:
            continue
        
        II = record.p_signal[:, 0]
        V5 = record.p_signal[:,1]
        ecg = np.stack([II, V5], axis=1)
 
        xann = wfdb.rdann(recordPath, 'xws')
        ann  = wfdb.rdann(recordPath, 'atr')

        # indices of ischemia events & normals
        pos_idx  = [i for i,s in zip(xann.sample, xann.symbol) if s in ischemia]
        norm_idx = [i for i,s in zip(ann.sample, ann.symbol)    if s == 'N']

        # positives
        stride = W // 4  # 75% overlap
        span = 4 * W     # how much of the episode to scan before/after

        for idx in pos_idx:
            for offset in range(-span//2, span//2, stride):
                start = idx + offset - W//2
                if 0 <= start and start + W <= len(ecg):
                    seg = ecg[start:start+W].astype(np.float32)
                    windows.append(seg.flatten())
                    labels.append(1)

        # normal samples, ~1:1 ratio
        k = len(pos_idx*16)
        if k > 0:
            chosen = random.sample(norm_idx, k=min(len(norm_idx), k))
            for idx in chosen:
                if idx+W <= len(ecg):
                    seg = ecg[idx:idx+W].astype(np.float32)
                    seg = seg.flatten()
                    windows.append(seg)
                    labels.append(0)

    X = np.stack(windows, axis=0)    # shape (n_samples, W)
    y = np.array(labels, dtype=np.int32).reshape(-1,1)
    return X, y

def MITShallow(hiddenStateSize, ETA, ):

    FOLDERDATAPATH = 'Data/MIT/mit-bih-st-change-database-1.0.0'
    X, Y = readMitMatrix(FOLDERDATAPATH, 6)
    validX, trainX, validY, trainY = splitDataNP(X,Y)
    Layers = createPipelineShallow(trainX, trainY, hiddenStateSize)
    train(validX, trainX, validY, trainY, Layers, f'MLP on MIT Shallow with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MIT(hiddenStateSize, ETA, ):

    FOLDERDATAPATH = 'Data/MIT/mit-bih-st-change-database-1.0.0'
    X, Y = readMitMatrix(FOLDERDATAPATH, 6)
    validX, trainX, validY, trainY = splitDataNP(X,Y)
    Layers = createPipelineDeep(trainX, trainY, hiddenStateSize)
    train(validX, trainX, validY, trainY, Layers, f'MLP on MIT with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def MITSKIP(hiddenStateSize, ETA, ):

    FOLDERDATAPATH = 'Data/MIT/mit-bih-st-change-database-1.0.0'
    X, Y = readMitMatrix(FOLDERDATAPATH, 6)
    validX, trainX, validY, trainY = splitDataNP(X,Y)
    Layers = createPipelineSkip(trainX, trainY, hiddenStateSize)
    train(validX, trainX, validY, trainY, Layers, f'MLP on MIT + Skip with eta = {ETA} and hidden = {hiddenStateSize}', ETA, )

def createPipelineShallow(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], trainY.shape[1], ADAM=True)
    L3 = TanhLayer()
    L4 = LogisticSigmoidLayer()
    L5 = LogLoss()

    Layers = [L1, L2, L3, L4, L5]
    return Layers

def createFullPipelineShallow(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], trainY.shape[1], ADAM=False)
    L3 = ReLULayer()
    L4 = LogisticSigmoidLayer()
    L5 = LogLoss()

    Layers = [L1, L2, L3, L4, L5]
    return Layers

def createPipelineDeep(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], hiddenStateSize, ADAM=True)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L5 = TanhLayer()
    L6 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L7 = TanhLayer()
    L8 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L9 = TanhLayer()
    L10 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L11 = TanhLayer()
    L12 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L13 = TanhLayer()
    L14 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L15 = TanhLayer()
    L16 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L17 = TanhLayer()
    L18 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True)
    L19 = TanhLayer()
    L20 = FullyConnectedLayer(hiddenStateSize, trainY.shape[1])
    L21 = LogisticSigmoidLayer()
    L22 = LogLoss()

    Layers = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22]
    return Layers

def createFullPipelineDeep(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], hiddenStateSize, ADAM=False)
    L3 = ReLULayer()
    L4 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L5 = ReLULayer()
    L6 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L7 = ReLULayer()
    L8 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L9 = ReLULayer()
    L10 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L11 = ReLULayer()
    L12 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L13 = ReLULayer()
    L14 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L15 = ReLULayer()
    L16 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L17 = ReLULayer()
    L18 = FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False)
    L19 = ReLULayer()
    L20 = FullyConnectedLayer(hiddenStateSize, trainY.shape[1])
    L21 = BatchNormLayer(trainY.shape[1])
    L22 = LogisticSigmoidLayer()
    L23 = LogLoss()

    Layers = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23]
    return Layers

def createPipelineSkip(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], hiddenStateSize)
    L3 = TanhLayer()
    L4 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-2, ADAM=True), BatchNormLayer(hiddenStateSize-2), TanhLayer()])
    hiddenStateSize -= 2
    L5 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True), BatchNormLayer(hiddenStateSize), TanhLayer()])
    L6 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-2, ADAM=True), BatchNormLayer(hiddenStateSize-2), TanhLayer()])
    hiddenStateSize -= 2
    L7 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True), BatchNormLayer(hiddenStateSize), TanhLayer()])
    L8 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-4, ADAM=True), BatchNormLayer(hiddenStateSize-4), TanhLayer()])
    hiddenStateSize -= 4
    L9 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True), BatchNormLayer(hiddenStateSize), TanhLayer()])
    L10 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-4, ADAM=True), BatchNormLayer(hiddenStateSize-4), TanhLayer()])
    hiddenStateSize -= 4
    L11 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=True), BatchNormLayer(hiddenStateSize), TanhLayer()])
    L12 = FullyConnectedLayer(hiddenStateSize, trainY.shape[1])
    L13= LogisticSigmoidLayer()
    L14 = LogLoss()

    Layers = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14]
    return Layers

def createFullPipelineSkip(trainX, trainY, hiddenStateSize):
    L1 = InputLayer(trainX)
    L2 = FullyConnectedLayer(trainX.shape[1], hiddenStateSize)
    L3 = ReLULayer()
    L4 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-2, ADAM=False), BatchNormLayer(hiddenStateSize-2), ReLULayer()])
    hiddenStateSize -= 2
    L5 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False), BatchNormLayer(hiddenStateSize), ReLULayer()])
    L6 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-2, ADAM=False), BatchNormLayer(hiddenStateSize-2), ReLULayer()])
    hiddenStateSize -= 2
    L7 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False), BatchNormLayer(hiddenStateSize), ReLULayer()])
    L8 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-4, ADAM=False), BatchNormLayer(hiddenStateSize-4), ReLULayer()])
    hiddenStateSize -= 4
    L9 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False), BatchNormLayer(hiddenStateSize), ReLULayer()])
    L10 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize-4, ADAM=False), BatchNormLayer(hiddenStateSize-4), ReLULayer()])
    hiddenStateSize -= 4
    L11 = MultiSkipLayer([FullyConnectedLayer(hiddenStateSize, hiddenStateSize, ADAM=False), BatchNormLayer(hiddenStateSize), ReLULayer()])
    L12 = FullyConnectedLayer(hiddenStateSize, trainY.shape[1])
    L13 = BatchNormLayer(trainY.shape[1])
    L14= LogisticSigmoidLayer()
    L15 = LogLoss()

    Layers = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15]
    return Layers

def runTests():
    results = []
    args = parseArguments()

    for hiddenStateSize in hiddenStateSizes:
        for ETA in ETAs:
                if args.command == 'ecg200':
                    print(f"Running ECG200 Block with ETA = {ETA} and Hidden = {hiddenStateSize}")
                    results.append(f"Shallow MLP200 at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200Shallow(hiddenStateSize, ETA, )}") ###MLP200
                    results.append(f"MLP200 at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200(hiddenStateSize, ETA, )}")
                    results.append(f"MLP200 with Skip at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200Skip(hiddenStateSize, ETA, )}")
                elif args.command == 'ecg5000':
                    print(f"Running ECG5000 Block with ETA = {ETA} and Hidden = {hiddenStateSize}")
                    results.append(f"Shallow MLP5000 at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Shallow(hiddenStateSize, ETA, )}") ###MLP5000
                    results.append(f"MLP5000 at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000(hiddenStateSize, ETA, )}")
                    results.append(f"MLP5000 with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Skip(hiddenStateSize, ETA, )}")
                elif args.command == 'ecg5000multiclass':
                    print(f"Running ECG5000 Block (Multiclass)")
                    results.append(f"MLP5000 Multiclass at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000MulticlassShallow(hiddenStateSize, ETA, )}") ###MLP5000 Multiclass
                    results.append(f"MLP5000 Multiclass at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Multiclass(hiddenStateSize, ETA, )}")
                    results.append(f"MLP5000 Multiclass with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000SkipMulticlass(hiddenStateSize, ETA, )}")
                elif args.command == 'mit':
                    print(f"Running MIT Block with ETA = {ETA} and Hidden = {hiddenStateSize}")
                    results.append(f"MIT at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MITShallow(hiddenStateSize, ETA, )}") ###MIT
                    results.append(f"MIT at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MIT(hiddenStateSize, ETA, )}")
                    results.append(f"MIT with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MITSKIP(hiddenStateSize, ETA, )}")
                elif args.command == 'full':
                    print(f"Running Full Block with ETA = {ETA} and Hidden = {hiddenStateSize}")
                    results.append(f"Full at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECGShallow(hiddenStateSize, ETA, )}") ###MIT
                    results.append(f"Full at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECG(hiddenStateSize, ETA, )}")
                    results.append(f"Full with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECGSkip(hiddenStateSize, ETA, )}")
                else:
                    print(f"Running All Blocks with ETA = {ETA} and Hidden = {hiddenStateSize}")
                    results.append(f"Shallow MLP200 at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200Shallow(hiddenStateSize, ETA, )}") ###MLP200
                    results.append(f"MLP200 at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200(hiddenStateSize, ETA, )}")
                    results.append(f"MLP200 with Skip at HSS {hiddenStateSize}, ETA {ETA}, Accuracy = {MLP200Skip(hiddenStateSize, ETA, )}")
                    
                    results.append(f"Shallow MLP5000 at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Shallow(hiddenStateSize, ETA, )}") ###MLP5000
                    results.append(f"MLP5000 at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000(hiddenStateSize, ETA, )}")
                    results.append(f"MLP5000 with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Skip(hiddenStateSize, ETA, )}")
                    
                    results.append(f"MLP5000 Multiclass at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000MulticlassShallow(hiddenStateSize, ETA, )}") ###MLP5000 Multiclass
                    results.append(f"MLP5000 Multiclass at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000Multiclass(hiddenStateSize, ETA, )}")
                    results.append(f"MLP5000 Multiclass with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MLP5000SkipMulticlass(hiddenStateSize, ETA, )}")
                    
                    results.append(f"MIT at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MITShallow(hiddenStateSize, ETA, )}") ###MIT
                    results.append(f"MIT at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MIT(hiddenStateSize, ETA, )}")
                    results.append(f"MIT with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {MITSKIP(hiddenStateSize, ETA, )}")

                    results.append(f"Full at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECGShallow(hiddenStateSize, ETA, )}") ###FULL
                    results.append(f"Full at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECG(hiddenStateSize, ETA, )}")
                    results.append(f"Full with Skip at HSS {hiddenStateSizes}, ETA {ETA}, Accuracy = {FullECGSkip(hiddenStateSize, ETA, )}")
                
    for result in results:
        print(result)

if __name__ == "__main__":
    runTests()
    exit(0)
