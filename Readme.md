## ECG Cardiac Ischemia Prediction Using MLP of Various Levels of Complexity

**Author**: David Everly  
**Language**: Python  
**Version**: 1 

---

# Problem Statement  
Acute Coronary Syndrome (ACS) demands rapid and accurate identification to improve patient outcomes. ECG interpretation remains a cornerstone of early ACS detection, but standard rule‑based algorithms and clinician review introduce delays and variability. We explore whether purely feed‑forward neural classifiers can reliably distinguish normal from ischemic single‑lead ECG cycles, with the aim of simplifying clinical workflows and reducing diagnostic latency.
  
# Description  
Main.py is a script which can run various experiments from command line. Each experiment trains and validates 3 MLPs on a specific dataset contained in the Data directory. Once the model is validated, an image the results is stored in the Images directory, one level above the code directory on the OS.  

# Purpose
To systematically evaluate MLP variants on single‑cycle ECG datasets of varying scale, determining trade‑offs between complexity, stability, and predictive performance for ischemia detection.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Results](#results)
- [Conclusion](#conslusion)
- [Future Work and Extension](#future-work-and-extension)
- [References](#references)
- [Contributing](#contributing)
- [Licenses](#licenses)

# Installation
Dependencies:   
numpy
pandas
matplotlib
wfdb
argparse

Install using:  
```bash
pip install -r requirements.txt  
```  

# Usage
Program is intended to be run using Unix-like terminal such as Linux, macOS Terminal (untested), or MINGW64 (Git Bash) on Windows.  

The script recognizes the following commands:  

```bash
python main.py [model]  
```

Models:  
ecg200  
ecg5000  
ecg5000multiclass  
mit  
full  

# Features  
Automatically creates, trains, and validates 3 MLP models on the specified dataset
- Displays training and validation performance over time
- Creates confusion matrix
- Calculates and displays model classification accuracy, sensitivity, and specificity

# Configuration  

## Data Sources
- **ECG200:** 200 one‑cycle tracings, normal vs. ischemic.  
- **ECG5000:** 5000 one‑cycle tracings, normal vs. abnormal (binary and multiclass subsets).
- **MIT-BIH:** 47 two-lead tracings over 30 minutes, normal vs. ischemia
- **Full ECG:** Al-Zaiti et al. dataset, 12-lead tracings 

*Note: MIT‑BIH and 12‑lead “Full ECG” data demonstrate poor convergence*

## Pre‑processing
- Z‑score normalization per cycle  
- 80/20 stratified train/validation split  

## Architectures
1. **Shallow MLP (5 layers):**
   - Input → Dense(64) → Tanh → Dense(32) → Tanh → Sigmoid

2. **Deep MLP (22 layers):**
   - 9× [Dense(32) → Tanh] → Dense(16) → Sigmoid

3. **Residual MLP (22 layers + shortcuts):**
   - Blocks of [Dense(32) → Tanh → BatchNorm] with identity shortcuts and funnel‑shaped sizes

## Training & Tuning
- **Optimizer:** Adam (learning rates swept over {1e‑3, 1e‑4, 1e‑5})  
- **Hidden sizes:** {8, 16, 32, 64} via grid search  
- Early stopping (patience = 10 epochs)

## Initialization
Xavier uniform for weights; biases initialized to zero.

# Results  

| Dataset     | Metric          | Shallow | Deep | Residual |
|-------------|-----------------|---------|------|----------|
| **ECG200**  | Accuracy        | 86 %    | 81 % | **91 %** |
| **ECG5000** | Binary Accuracy | 98 %    | 98 % | **98 %** |
| **ECG5000** | Multiclass Acc. | 92 %    | 95 % | **95 %** |

- **ECG200:** Residual MLP led (91 %), showing depth+shortcuts help on small/noisy samples.  
- **ECG5000:** All reached ∼98 % binary accuracy; deeper models improved multiclass by 3 %.

### ECG200 Results

| Shallow Model | Shallow CM | Deep Model | Deep CM |
|:-------------:|:----------:|:----------:|:-------:|
| [![Shallow](/FinalModels/mlp200/best%20model/mlp200_shallow_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp200/best%20model/mlp200_shallow_with_eta_0_0001_and_hidden_32.png) | [![Shallow CM](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_shallow_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_shallow_with_eta_0_0001_and_hidden_32_confusion.png) | [![Deep](/FinalModels/mlp200/best%20model/mlp200_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp200/best%20model/mlp200_with_eta_0_0001_and_hidden_32.png) | [![Deep CM](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_with_eta_0_0001_and_hidden_32_confusion.png) |

| Deep + Skip | Deep+Skip CM | 5000 Shallow | 5000 Shallow CM |
|:-----------:|:------------:|:------------:|:---------------:|
| [![Skip](/FinalModels/mlp200/best%20model/mlp200_skip_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp200/best%20model/mlp200_skip_with_eta_0_0001_and_hidden_32.png) | [![Skip CM](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_skip_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp200/best%20model/confusion_matrix_mlp200_skip_with_eta_0_0001_and_hidden_32_confusion.png) | [![5000 Shallow](/FinalModels/mlp5000/Best/mlp5000_shallow_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp5000/Best/mlp5000_shallow_with_eta_0_0001_and_hidden_32.png) | [![5000 Shallow CM](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_shallow_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_shallow_with_eta_0_0001_and_hidden_32_confusion.png) |

| 5000 Deep | 5000 Deep CM | 5000 Skip | 5000 Skip CM |
|:---------:|:------------:|:---------:|:-----------:|
| [![5000 Deep](/FinalModels/mlp5000/mlp5000_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp5000/mlp5000_with_eta_0_0001_and_hidden_32.png) | [![5000 Deep CM](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_with_eta_0_0001_and_hidden_32_confusion.png) | [![5000 Skip](/FinalModels/mlp5000/Best/mlp5000_skip_with_eta_0_0001_and_hidden_32.png)](/FinalModels/mlp5000/Best/mlp5000_skip_with_eta_0_0001_and_hidden_32.png) | [![5000 Skip CM](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_skip_with_eta_0_0001_and_hidden_32_confusion.png)](/FinalModels/mlp5000/Best/confusion_matrix_mlp5000_skip_with_eta_0_0001_and_hidden_32_confusion.png) |


# Conclusion
Residual connections significantly boost MLP performance on small, noisy ECG cycles. For large datasets, shallow networks suffice for binary tasks, but deep/residual models yield multiclass gains. Pure MLPs struggle with multi‑lead or long sequences, suggesting CNNs or Transformers for richer ECG representations.

# Future Work and Extension  
- Evaluate CNNs and Transformers on single‑cycle and full‑lead ECGs  
- Incorporate RNNs or attention mechanisms for multi‑beat/continuous monitoring  
- Conduct prospective clinical validation against cardiologist readings  

# References  
1. Al‑Zaiti et al., 2023. Machine learning for ECG diagnosis and risk stratification… _Nature Medicine_.  
2. Dau et al., 2019. The UCR Time Series Classification Archive. UCR.  
3. Goldberger et al., 2000. PhysioBank, PhysioToolkit, and PhysioNet. _Circulation_.  
4. Institute of Medicine, 2000. _To err is human: Building a safer health system_.  
5. Moody & Mark, 2001. The impact of the MIT‑BIH Arrhythmia Database. _IEEE EMBS Mag_.  
6. Xiong et al., 2022. Deep Learning for Detecting and Locating Myocardial Infarction… _Frontiers in Cardiovascular Medicine_.  

# Contributing  
No external parties contributed to this project.  

# Licenses  
None

<a href="https://www.dmeverly.com/completedprojects/cardiac-ischemia/" style="display: block; text-align:right;" target = "_blank">  Project Overview -> </a>  