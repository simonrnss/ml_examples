'''
Bagging example

Example python code that demonstrates a very simple bagging ensemble -- a random forest of stumps
(a stump is a decision tree with one layer).
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
def sub_sample(train_X: np.ndarray, train_y: np.ndarray, n_sample: int):
    '''Sample n_sample rows from train_X and train_y with replacement'''
    idx = np.arange(len(train_X))
    chosen_idx = np.random.choice(idx, size=n_sample, replace=True)
    return train_X[chosen_idx, :], train_y[chosen_idx]

# %%
d_trees = []
n_trees = 100
for i in range(n_trees):
    xx, yy = sub_sample(train_X, train_y, 500)
    d_tree = DecisionTreeClassifier(max_depth=1)
    d_tree.fit(xx, yy)
    d_trees.append(d_tree)
# %%
# Plot a single d-tree
idx = 1
plot_dt(train_X, train_y, [d_trees[idx]], [1], hard=False)

# %% plot the first N
n_tree_plot = 2
plot_dt(train_X, train_y, d_trees[:n_tree_plot], [1/n_tree_plot for i in range(n_tree_plot)], hard=False)
# %%
plot_dt(train_X, train_y, d_trees, [1/n_trees for i in range(n_trees)], hard=False)


# %%
# Auc versus number trees
from sklearn.metrics import roc_auc_score
all_auc = []
test_X, test_y = sample_data(n_samples=1000)
preds = np.zeros_like(test_y, float)
for i, tree in enumerate(d_trees):
    preds += tree.predict(test_X)
    auc_score = roc_auc_score(test_y, preds)
    all_auc.append(auc_score)
plt.plot(range(n_trees), all_auc)
# %%
all_auc[-1]
# %%
