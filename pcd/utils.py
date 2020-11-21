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

def find_coords(rad, sep, init_angle=0, fin_angle=360):
    angular_range = fin_angle - init_angle
    npoints = (np.deg2rad(angular_range) * rad) / sep  # (2*np.pi*rad)/sep
    ang_step = 360 / npoints  # 360/npoints
    x = []
    y = []
    for i in range(int(npoints)):
        newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
        newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
        x.append(newx)
        y.append(newy)
    return np.array([np.array(y), np.array(x)])