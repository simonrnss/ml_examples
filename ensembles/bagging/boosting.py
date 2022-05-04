'''
boosting
'''
# %% Imports
from typing import Tuple, List
import numpy as np
import pylab as plt
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline

# %% Utility method to generate non-linear data
def sample_data(n_samples: int=100, n_dim: int=2, threshold: float=2.0) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generates n_samples of n_dim dimensions from independent standardised Gaussians
    The labels (y) is +1 or -1 depending on whether the sum of squares of the
    value is greater than threshold.
    '''
    X = np.random.normal(size=(n_samples, n_dim))
    temp = (X**2).sum(axis=1) > threshold
    y = 2 * temp - 1
    return X, y
# %% Sample and plot some training data
train_X, train_y = sample_data(n_samples=1000)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
# %%
def plot_dt(train_X: np.ndarray, train_y: np.ndarray, d_trees:List[DecisionTreeClassifier], alpha_vals: List[float], hard: bool=False):
    '''
    A method to plot the predictions from a tree ensemble. The ensemble is
    a list of DecisionTreeClassifier objects with weights given by alpha_vals
    hard = True means the final prediction is thresholded to +-1
    '''
    min_x = min(train_X[:, 0])
    min_x -= 0.05 * min_x
    max_x = max(train_X[:, 0])
    max_x += 0.05 * max_x
    min_y = min(train_X[:, 1])
    min_y -= 0.05 * min_y
    max_y = max(train_X[:, 1])
    max_y += 0.05 * max_y

    num = 100 # Number of grid points
    
    x_grid, y_grid = np.meshgrid(
        np.linspace(min_x, max_x, num=num),
        np.linspace(min_y, max_y, num=num)
    )
    test_data = np.hstack((
        x_grid.reshape((num**2, 1)),
        y_grid.reshape((num**2, 1))
    ))

    # Compute predictions
    preds = np.zeros((num**2,), float)
    for i, d_tree in enumerate(d_trees):
        preds = preds + alpha_vals[i] * d_tree.predict(test_data)
    if hard:
        preds = np.sign(preds)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=preds, alpha=0.75, s=4)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
    plt.colorbar()

# %%
sample_weights = np.ones_like(train_y, float)
sample_weights /= sample_weights.sum()
n_trees = 10
d_trees = []
alpha_vals = []
for i in range(n_trees):
    d_tree = DecisionTreeClassifier(max_depth=1)
    d_tree.fit(train_X, train_y, sample_weight = sample_weights)
    
# %%

# %%
