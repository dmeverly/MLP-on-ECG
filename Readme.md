# ECG Ischemia Prediction (MLP Study)

**Author**: David Everly  
**Language**: Python  
**Domain**: Clinical Signal Analysis / Applied Machine Learning  
**Status**: Exploratory

---

## Description

This project explores the effectiveness and limitations of multilayer perceptron (MLP) architectures for ECG-based ischemia classification across datasets of increasing clinical complexity.

Rather than aiming for state-of-the-art performance, the goal is to understand **where and why simpler feedforward models break down** when applied to richer, noisier clinical signals. The findings from this work informed later architectural choices, including the transition to convolutional models for waveform analysis.

---

## Scope and Intent

This project is intentionally exploratory.

It focuses on:
- empirical comparison of MLP depth and structure
- convergence behavior across datasets
- identifying architectural mismatch between model class and signal complexity

It does **not** attempt:
- clinical deployment
- production-grade performance
- model interpretability beyond basic analysis

---

## System Characteristics

- **Model Families**  
  Evaluates shallow, deep, and residual MLP architectures.

- **Datasets**  
  Experiments conducted across multiple ECG datasets, including:
  - ECG200
  - ECG5000
  - MIT-BIH–derived data
  - higher-dimensional ECG representations

- **Training Discipline**  
  Includes early stopping, hyperparameter sweeps, and repeated trials to assess convergence stability.

- **Evaluation Metrics**  
  Accuracy, sensitivity, specificity, and confusion matrices are reported to understand class-level behavior.

---

## Key Findings

- MLPs can perform adequately on simplified or low-dimensional ECG datasets.
- As signal richness and dataset complexity increase, MLPs exhibit:
  - unstable convergence
  - poor generalization
  - sensitivity to hyperparameter tuning
- Architectural expressiveness, rather than training duration, becomes the limiting factor.

These results motivated the transition to convolutional architectures in subsequent projects.

---

## What This Project Demonstrates

- Empirical evaluation of model–data mismatch  
- Willingness to conclude when an approach is insufficient  
- Methodical experimentation across datasets and architectures  
- Judgment in evolving system design based on observed failure modes

---

## Limitations and Non-Goals

- This project does not include interpretability tooling such as saliency analysis
- No attempt is made to handle real-time or streaming ECG data
- Results are dataset-specific and not clinically validated

---

## Relationship to Other Projects

This work directly informed later projects, including:

- **CNN Cardiac Rhythm Classification**, which applies convolutional architectures better suited to temporal signal structure and interpretability.

---

## Disclaimer

This project was developed independently on personal time and is not affiliated with or endorsed by any employer.  
All data used is publicly available.  
This work reflects exploratory experimentation rather than deployable clinical systems.
