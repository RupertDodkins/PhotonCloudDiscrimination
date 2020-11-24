import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np
import pickle

from pcd.config.config import config

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

def get_range_inds(start, end, allsteps):
    if start < 0:
        start = allsteps+start
    if end < 0:
        end = allsteps + end
    return start, end

def confusion_matrix(false_neg, true_pos, true_neg, false_pos, tot_neg, tot_pos):
    if tot_pos == 0.0:
        conf = ('      +------+\n'
                '     1| %.2f |\n'
                'Pred -+------+\n'
                '     0| %.2f |\n'
                '      +------+\n'
                '         0    \n'
                '           True' % (false_pos / tot_neg,
                                     true_neg / tot_neg))

    else:
        conf = ('      +------+------+\n'
                '     1| %.2f | %.2f |\n'
                'Pred -+------+------+\n'
                '     0| %.2f | %.2f |\n'
                '      +------+------+\n'
                '         0   |  1\n'
                '           True' % (false_pos / tot_neg, true_pos / tot_pos,
                                     true_neg / tot_neg, false_neg / tot_pos))
    print('true_pos: %f' % (true_pos))
    print('true_neg: %f' % (true_neg))
    print('false_pos: %f' % (false_pos))
    print('false_neg: %f' % (false_neg))
    return conf

def get_metric_distributions(true_label, pred_label, sum=True):
    true_neg = np.logical_and(true_label == 0, np.round(pred_label) == 0)  # round just in case the pred_val is in mean mode
    true_pos = np.logical_and(true_label == 1, np.round(pred_label) == 1)
    false_neg = np.logical_and(true_label == 1, np.round(pred_label) == 0)
    false_pos = np.logical_and(true_label == 0, np.round(pred_label) == 1)

    metrics = [true_pos, false_neg, false_pos, true_neg]

    if sum:
        metrics = [int(np.sum(metric)) for metric in metrics]

    return metrics

def initialize_axes(metric_types):
    fig, axes = init_grid(rows=2, cols=int(np.ceil(len(metric_types)/2)), figsize=(16, 9))
    axes = axes.flatten()
    for ax, metric_type in zip(axes, metric_types):
        ax.set_title(metric_type)
    return axes

def load_meta(kind='pt_outputs', amount=-1):
    alldata = []
    with open(config['train'][kind], 'rb') as handle:
        while True:
            try:
                if amount == 1:
                    alldata = [pickle.load(handle)]
                else:
                    alldata.append(pickle.load(handle))
            except EOFError:
                break

    return alldata
