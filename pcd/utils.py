import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np

def init_grid(rows, cols):
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(12, rows*3), constrained_layout=True)
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))
    axes = np.array(axes).reshape(rows, cols)
    plt.tight_layout()

    return fig, axes
