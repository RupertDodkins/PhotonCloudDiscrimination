import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np

def init_grid(rows, cols, figsize=None):
    gs = gridspec.GridSpec(rows, cols)
    if figsize is None:
        figsize = (12, 3*cols)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))
    axes = np.array(axes).reshape(rows, cols)
    plt.tight_layout()

    return fig, axes
