import numpy as np
import matplotlib.pyplot as plt


def plotcluster2(X, labels, ax=None):
    """
    Plot 2D embedding with colored cluster labels (MATLAB plotcluster2 style).

    Parameters are:
    - X: N by D matrix. Each row in X represents an observation.
    - labels: clustering annotations for data X.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if ax is None:
        _, ax = plt.subplots()

    if not (np.all(labels == np.floor(labels)) and np.all(labels >= 0)):
        print("WARNING: clustering annotation must be a non-negative integer!")
        return ax

    colors = np.array(
        [
            [31, 119, 179],
            [251, 130, 20],
            [43, 159, 46],
            [210, 33, 33],
            [143, 99, 187],
            [140, 87, 76],
            [255, 116, 192],
            [200, 200, 200],
            [184, 187, 29],
            [30, 191, 208],
            [218, 165, 32],
            [65, 105, 225],
            [255, 99, 71],
            [147, 112, 219],
            [255, 215, 0],
            [50, 205, 50],
            [174, 199, 232],
            [255, 187, 120],
            [152, 223, 138],
            [255, 152, 150],
            [196, 177, 213],
            [196, 155, 147],
            [219, 219, 141],
            [135, 206, 235],
            [255, 165, 0],
            [144, 238, 144],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [160, 0, 160],
            [12, 128, 144],
            [255, 69, 0],
            [140, 86, 75],
            [160, 82, 45],
            [0, 139, 139],
            [175, 238, 238],
            [233, 150, 122],
            [143, 188, 143],
            [106, 90, 205],
            [60, 179, 113],
            [220, 20, 60],
            [65, 105, 225],
            [147, 112, 219],
            [20, 206, 209],
        ],
        dtype=float,
    ) / 255.0

    for i in range(len(labels)):
        lab = int(labels[i])
        if lab == 0:
            ax.plot(
                X[i, 0],
                X[i, 1],
                "o",
                markerfacecolor=(0, 0, 0),
                markeredgecolor=(0, 0, 0),
                markersize=4,
            )
        elif lab <= 40:
            color = colors[lab - 1]
            ax.plot(
                X[i, 0],
                X[i, 1],
                "o",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4,
            )
        else:
            color = colors[40 + ((lab - 41) % 7)]
            ax.plot(
                X[i, 0],
                X[i, 1],
                "o",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4,
            )

    return ax
