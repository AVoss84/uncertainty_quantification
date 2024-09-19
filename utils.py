from typing import Dict, List, Text, Optional, Any, Callable, Union, Tuple
import numpy as np
from numpy.linalg import pinv as inv        # Moore/Penrose pseudo inverse
from numpy.random import normal
from math import cos, pi
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
from scipy.stats import multivariate_t, multivariate_normal, invgamma
from sklearn.model_selection import train_test_split


def print_image_by_id(id: int, images: np.ndarray, labels: np.array = None) -> None:
    """
    Print image by id
    :param id: id of the image
    :return: None
    """
    plt.figure(figsize=(2, 2))  # Smaller figure size for individual images
    plt.imshow(images[id].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Image {id}")
    if labels is not None:
        plt.title(f"Image {id} - Label: {labels[id]}")
    else:
        plt.title(f"Image {id}")
    plt.axis('off')
    plt.show()


def plot_scores(n, alphas, scores, quantiles, **kwargs):
    
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    plt.figure(figsize=(6, 4))
    plt.hist(scores, **kwargs)
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=400,
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}"
        )
    plt.title("Distribution of conformity scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()


def posterior_betas(X: np.ndarray, y: np.array, beta_0: np.array, M0: np.array, s0: float = 1, nu0: float = 5)-> tuple:    
    """Compute parameters of posterior distribution of beta coefficients

    Args:
        X (np.ndarray): _description_
        y (np.array): _description_
        beta_0 (np.array): _description_
        M0 (np.array): _description_
        s0 (float, optional): _description_. Defaults to 1.
        nu0 (float, optional): _description_. Defaults to 5.

    Returns:
        tuple: _description_
    """
    assert isinstance(X, np.ndarray), 'X must be a numpy array'
    assert isinstance(y, np.ndarray), 'y must be a numpy array'
    n = X.shape[0]
    M_star = M0 + (X.T @ X)
    beta_star = inv(M_star) @ (M0 @ beta_0 + X.T @ y)      # Bayes estimate
    Mx = np.eye(n, n) - X @ inv(X.T @ X) @ X.T             # residual maker
    s_ols = y.T @ Mx @ y                                   # sample variance
    beta_ols = inv(X.T @ X) @ X.T @ y                      # OLS estimate
    s_star = s0 + s_ols + (beta_0.T @ inv(M0) @ beta_0) + (beta_ols.T @ X.T @ X @ beta_ols) - beta_star.T @ inv(M_star) @ beta_star
    nu_star = nu0 + n                                      # posterior d.o.f.
    varcov_beta = (s_star/(nu_star - 2))*inv(M_star)       # posterior covariance matrix
    return beta_star, beta_ols, varcov_beta, M_star, s_star, nu_star


class Predictive:
    """Predictive distribution of y_new given X_new and y"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print('--- Predictive distribution of y_new given X_new and y ---')
         
    def posterior_predictive(self, X: np.ndarray, X_new: np.ndarray, y: np.array, **kwargs) -> np.ndarray:
        """Posterior predictive distribution of y_new

        Args:
            X (np.ndarray): _description_
            X_new (np.ndarray): _description_
            y (np.array): _description_

        Returns:
            np.ndarray: _description_
        """
        assert isinstance(X, np.ndarray), 'X must be a numpy array'
        assert isinstance(y, np.ndarray), 'y must be a numpy array'
        assert isinstance(X_new, np.ndarray), 'X_new must be a numpy array'
        k = X.shape[1]
        n_new = X_new.reshape(-1,k).shape[0]
        beta_star, beta_ols, varcov_beta, M_star, s_star, self.nu_star = posterior_betas(X=X, y=y, **kwargs)
        self.predictive_mean = X_new @ beta_star          # predictive mean
        self.predictive_varcov = (s_star/(self.nu_star - 2))*(np.eye(n_new, n_new) + X_new @ inv(M_star) @ X_new.T)     # predictive covariance matrix
        return self.predictive_mean, self.predictive_varcov

    def draw(self, size: int = 1)-> np.ndarray:
        """Draw from posterior predictive distribution

        Args:
            size (int, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """
        self.mvt = multivariate_t(loc=self.predictive_mean, shape=self.predictive_varcov, df=self.nu_star)
        return self.mvt.rvs(size=size)


def phi(x: np.array)-> np.ndarray:
    """Matrix of cubic basis functions"""
    return np.concatenate( (x*0+1, x, x**2, x**3), axis=1 )   # bias term incl.!


def polyDGP(n: int = 5*10**3, sigma2: float = 49, w0: np.ndarray = np.array([0, -5, 0, 1]), seed: int = None)-> tuple:
    """Cubic Gaussian Data Generating Process

    Args:
        n (int, optional): _description_. Defaults to 5*10**3.
        sigma2 (float, optional): _description_. Defaults to 49.
        w0 (np.ndarray, optional): _description_. Defaults to np.array([0, -5, 0, 1]).
        seed (int, optional): _description_. Defaults to None.

    Returns:
        tuple: _description_
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(low=-5, high=5, size=(n, 1))
    if w0 is None:
        w = np.random.normal(loc=w0, size=w0.shape)
    else:
        w = w0
    y = phi(X) @ w + np.random.normal(0, scale=np.sqrt(sigma2), size=X.shape[0])       # function plus homoscedastic gaussian noise
    return y, w, phi(X)


def cycle(N: int, nof_cycles: int = 1)-> np.array:
    """Cosine function with nof_cycles cycles

    Args:
        N (int): _description_
        nof_cycles (int, optional): _description_. Defaults to 1.

    Returns:
        np.array: _description_
    """
    return np.cos(2*pi*np.arange(0,N)*nof_cycles/N)


def simAR1(N: int, phi: float, sigma: float, const: float = 0, burn: int = 100)-> np.array:
    """Simulate Gausssian AR(1) process

    Args:
        N (int): _description_
        phi (float): _description_
        sigma (float): _description_
        const (float, optional): _description_. Defaults to 0.
        burn (int, optional): _description_. Defaults to 100.

    Returns:
        np.array: _description_
    """
    y = np.zeros((N+burn))
    for t in range(N+burn-1):
        y[t+1] = const + phi*y[t] + normal(scale = sigma, size=1)
    return y[burn:]


def embed(data: pd.DataFrame, lags: int = 1, include_original: bool = True, dropnan: bool = True)-> pd.DataFrame:   
    """
    Create lagged versions of input data.

    Parameters:
    data (pd.DataFrame): DataFrame containing time series data.
    lags (int): Number of lagged versions to create.
    include_original (bool): Whether to include the original data (lag 0) in the output. Default is True.
    dropnan (bool): Whether to drop rows with NaN values. Default is True.

    Returns:
    pd.DataFrame: DataFrame containing the original and lagged data.

    Raises:
    ValueError: If `data` is not a pandas DataFrame or `lags` is not a non-negative integer.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("`lags` must be a non-negative integer")

    if data.empty or lags >= len(data):
        raise ValueError("Data does not have enough rows to create requested lags")

    col_names = data.columns
    lagged_data = {f'{col}_lag{i}': data[col].shift(i) for i in range(lags + 1) for col in col_names}

    if not include_original:
        lagged_data = {name: col for name, col in lagged_data.items() if '_lag0' not in name}

    result = pd.DataFrame(lagged_data)

    if dropnan:
        result.dropna(inplace=True)
    return result


def plot_1d_data(
    X_train,
    y_train,
    X_test,
    y_test,
    y_sigma,
    y_pred,
    y_pred_low,
    y_pred_up,
    ax=None,
    title=None
):
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axes if none provided

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="Training data")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_sigma, color="gray", ls="--")                 # lower bound of true confidence intervals: y - lambda*sigma
    ax.plot(X_test, y_test + y_sigma, color="gray", ls="--")                 # upper bound of true confidence intervals: y + lambda*sigma
    ax.plot(X_test, y_pred, color="blue", alpha=0.5, label="Prediction intervals")
    
    if title is not None:
        ax.set_title(title)
    ax.legend()


def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x*np.sin(x)

def get_1d_data_with_constant_noise(funct, min_x, max_x, n_samples, noise, seed=59):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    np.random.seed(seed)
    X_train = np.linspace(min_x, max_x, n_samples)      # training data
    np.random.shuffle(X_train)                  # shuffle the data
    X_test = np.linspace(min_x, max_x, n_samples*5)   # testing data

    # True unknown response function
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)    
    # add homoscedastic gaussian noise to regression function 
    y_train += np.random.normal(loc=0, scale=noise, size=y_train.shape[0])      # add jitter
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh


def sample_cond_post_betas_sigma2(X: np.array, y: np.array, sigma2: np.array = np.array([1]), size: int = 1000)-> np.array:
    """Draws from the conditional posterior distribution of the betas given sigma2.
    Assuming a normal prior for the betas, noninformative prior for sigma2 
    and a normal sample density for y given X and parameters.

    Args:
        X (np.array): _description_
        y (np.array): _description_
        sigma2 (np.array, optional): _description_. Defaults to np.array([1]).
        size (int, optional): _description_. Defaults to 1000.

    Returns:
        np.array: _description_
    """
    beta_ols = inv(X.T @ X) @ X.T @ y      # OLS
    var_beta = inv(X.T @ X)                # OLS variance
    post_beta_sig = multivariate_normal(mean=beta_ols, cov=var_beta*sigma2)
    return post_beta_sig.rvs(size=size)

def sample_marg_post_sigma2(X: np.array, y: np.array, size: int = 1000)-> np.array:
    """Draws from the marginal posterior distribution of sigma2.

    Args:
        X (np.array): _description_
        y (np.array): _description_
        size (int, optional): _description_. Defaults to 1000.

    Returns:
        np.array: _description_
    """
    n, k = X.shape                                         
    Mx = np.eye(n, n) - X @ inv(X.T @ X) @ X.T             
    s2_ols = (y.T @ Mx @ y)/(n-k)           # sample variance
    df = n-k                                # degrees of freedom for the chi-square distribution
    scale = 0.5*s2_ols                      # scale parameter (scale = 1/2 for chi-square)
    return invgamma.rvs(a=df/2, scale=scale, size=size)


def samples_joint_beta_sigma2(X: np.array, y: np.array, size: int = 1000)-> np.array:
    """Draws from the joint posterior distribution of beta and sigma2.

    Args:
        X (np.array): _description_
        y (np.array): _description_
        size (int, optional): _description_. Defaults to 1000.

    Returns:
        np.array: _description_
    """
    samples_sigma2 = sample_marg_post_sigma2(X, y, size=size)                   # Draw from marginal posterior distribution of sigma2
    samples_betas = sample_cond_post_betas_sigma2(X, y, sigma2=1.0, size=1)     # initial value for the betas

    # Draw from the conditional posterior distribution of the betas given sigma2
    for sig2 in samples_sigma2:
        samples_betas = np.vstack((samples_betas, sample_cond_post_betas_sigma2(X, y, sigma2=sig2, size=1)))
    
    samples_joint = np.concatenate([samples_betas[1:], samples_sigma2.reshape(-1, 1)], axis=1)
    return samples_joint