# Uncertainty Quantification in Machine Learning

This repository provides code examples for an introduction to **uncertainty quantification (UQ)** in supervised machine learning, with an emphasis on:

- **Conformal Prediction/Inference** — a distribution-free framework for constructing prediction sets with guaranteed coverage
- **Bayesian Methods** — parametric approaches for quantifying uncertainty through posterior distributions

## 📁 Repository Structure

### Conformal Prediction Notebooks

| File | Description |
|------|-------------|
| [Introduction_Conformal_Prediction.ipynb](Introduction_Conformal_Prediction.ipynb) | Introduction to conformal prediction for classification using the Dry Bean dataset (UCI). Covers data splitting, calibration sets, and prediction set construction. |
| [conformal_classifier_MNIST.ipynb](conformal_classifier_MNIST.ipynb) | Conformal inference for image classification on Fashion-MNIST dataset using Random Forest and SVM classifiers. |
| [conformal_regression_ex.ipynb](conformal_regression_ex.ipynb) | Conformal inference for regression problems. Demonstrates aleatoric uncertainty estimation using MAPIE library with quantile regression and coverage score evaluation. |

### Bayesian Inference Notebooks

| File | Description |
|------|-------------|
| [bayesian_example.ipynb](bayesian_example.ipynb) | Bayesian prediction for autoregressive AR(1) processes. Demonstrates posterior inference for time series forecasting with uncertainty bands. |
| [bayesian_fnn.ipynb](bayesian_fnn.ipynb) | Bayesian linear regression using Pyro with variational inference (SVI) and automatic guides. |
| [bayesina_nn_pyro.ipynb](bayesina_nn_pyro.ipynb) | Bayesian Neural Networks (BNN) using Pyro. Implements a neural network with probabilistic weights for regression with uncertainty estimation. |
| [bayesUC_pymc3.ipynb](bayesUC_pymc3.ipynb) | Bayesian Unobserved Components (UC) model using PyMC3. Example with US Consumer Price Index data for time series decomposition. |

### Utility Files

| File | Description |
|------|-------------|
| [utils.py](utils.py) | Helper functions including data simulation (`simAR1`, `cycle`), embedding utilities (`embed`), Bayesian posterior computations (`posterior_betas`, `Predictive`), and visualization tools (`plot_scores`, `plot_1d_data`). |
| [requirements.txt](requirements.txt) | Python package dependencies |

## 🚀 Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv env_uq
source env_uq/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Additional dependencies for Bayesian notebooks

For the Pyro-based notebooks:
```bash
pip install pyro-ppl torch
```

For the PyMC3 notebook:
```bash
pip install pymc3 statsmodels pandas-datareader
```

## 📚 Key Concepts

### Conformal Prediction
A distribution-free method that wraps any machine learning model to produce prediction sets with valid coverage guarantees. The key idea is to use a calibration set to determine prediction thresholds.

### Bayesian Uncertainty Quantification
Provides uncertainty estimates through posterior distributions over model parameters, enabling:
- Credible intervals for predictions
- Model averaging
- Principled handling of parameter uncertainty

## 🛠️ Libraries Used

- **[MAPIE](https://mapie.readthedocs.io/)** - Model Agnostic Prediction Interval Estimator for conformal prediction
- **[Pyro](https://pyro.ai/)** - Probabilistic programming with PyTorch
- **[PyMC3](https://docs.pymc.io/)** - Bayesian modeling and probabilistic machine learning
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning models and utilities
