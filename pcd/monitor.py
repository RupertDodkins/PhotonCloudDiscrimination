""" Monitors the predictions of a training model by reading the pkl cache """

import os
import argparse
import pickle
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import numpy as np
import time
from config.medis_params import mp, ap
from config.config import config
import utils

def load_meta():
    alldata = []
    with open(config['train']['ml_meta'], 'rb') as handle:
        while True:
            try:
                alldata.append(pickle.load(handle))
            except EOFError:
                break
    return alldata

def get_metrics(cur_seg, pred_seg_res, include_true_neg=True):
    true_neg = np.logical_and(cur_seg == 0, pred_seg_res == 0)
    true_pos = np.logical_and(cur_seg == 1, pred_seg_res == 1)
    false_neg = np.logical_and(cur_seg == 1, pred_seg_res == 0)
    false_pos = np.logical_and(cur_seg == 0, pred_seg_res == 1)
    if include_true_neg:
        metrics = [true_pos, false_neg, false_pos, true_neg]
        scores = np.array([np.sum(true_pos), np.sum(false_pos)]) / np.sum(cur_seg == 1)
        print(scores)
        scores = np.array([np.sum(true_neg), np.sum(false_neg)]) / np.sum(cur_seg == 0)
        print(scores)
    else:
        metrics = [true_pos, false_neg, false_pos]

    return metrics

def three_d_scatter(cur_data, metrics):
    fig = plt.figure(figsize=(12, 9))
    colors = ['green', 'orange', 'purple', 'blue', ]
    ax = fig.add_subplot(111, projection='3d')
    for metric, c in zip(metrics, colors[:len(metrics)]):
        red_data = cur_data[metric]
        ax.scatter(red_data[:, 3], red_data[:, 1], red_data[:, 2], c=c, marker='o', s=2)  # , marker=pids[0])
    ax.view_init(elev=10., azim=-10)
    plt.show()

def cloud(epoch=-1):
    alldata = load_meta()
    cur_seg, pred_seg_res, cur_data = alldata[epoch]
    metrics = get_metrics(cur_seg, pred_seg_res)
    three_d_scatter(cur_data, metrics)

def confusion_matrix(false_neg, true_pos, true_neg, false_pos, tot_neg, tot_pos):
    return ('      +------+------+\n'
            '     1| %.2f | %.2f |\n'
            'Pred -+------+------+\n'
            '     0| %.2f | %.2f |\n'
            '      +------+------+\n'
            '         0   |  1\n'
            '           True' % (false_pos/tot_pos, true_pos/tot_pos,
                                 true_neg/tot_neg, false_neg/tot_neg))

def plot_metric_tesseracts(start=0, end=-1, include_true_neg=True):
    """ Shows the net predictions on the cloud as a series of 2d histograms in 4x4 grid of form

         xy | py | ty | pt
    TP |
    FP |
    TN |
    FN |


    """
    assert end != 0

    alldata = load_meta()
    allsteps = len(alldata)

    # set axes
    plt.ion()
    plt.show(block=True)
    rows = 4 if include_true_neg else 3
    fig, axes = utils.init_grid(rows=rows, cols=config['dimensions'], figsize=(12,9))

    plt.figtext(.07,.8,'True Planet', fontsize=18, ha='left')
    plt.figtext(.07,.56,'Missed Planet', fontsize=18, ha='left')
    plt.figtext(.07,.32,'True Star', fontsize=18, ha='left')
    plt.figtext(.07,.08,'Missed Star', fontsize=18, ha='left')

    for i in range(rows):
        axes[i,0].set_ylabel('Y')
        axes[i,1].set_ylabel('Y')
        axes[i,2].set_ylabel('Y')
        axes[i,3].set_ylabel('Time')

    axes[-1,0].set_xlabel('X')
    axes[-1,1].set_xlabel('Phase')
    axes[-1,2].set_xlabel('Time')
    axes[-1,3].set_xlabel('Phase')

    ims = []
    if start < 0:
        start = allsteps+start
    if end < 0:
        end = allsteps + end

    time.sleep(5)

    for epoch in range(start, end+1):
        print(epoch)
        fig.suptitle(f'step {epoch}/{allsteps-1}', fontsize=16)

        if len(ims) > 0:
            [im.remove() for im in ims]
            ims = []

        cur_seg, pred_seg_res, cur_data = alldata[epoch]

        metrics = get_metrics(cur_seg, pred_seg_res, include_true_neg)
        true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics[0])), int(np.sum(metrics[1])), \
                                                   int(np.sum(metrics[2])), int(np.sum(metrics[3])),

        print(true_pos, false_neg, false_pos, true_neg)
        try:
            print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg+false_neg, true_pos+false_pos))
        except ZeroDivisionError:
            pass

        metrics[2], metrics[3] = metrics[3], metrics[2]

        for row, metric in enumerate(metrics):
            red_data = cur_data[metric]

            bins = [range(mp.array_size[0]), range(mp.array_size[1])]
            H, _, _ = np.histogram2d(red_data[:, 2], red_data[:, 3], bins=bins)
            ims.append(axes[row, 0].imshow(H, norm=LogNorm(), aspect='auto',
                                           extent=[min(bins[1]),max(bins[1]),min(bins[0]),max(bins[0])]))

            bins = [range(mp.array_size[0]), np.linspace(-120, 0, 50)]
            H, _, _ = np.histogram2d(red_data[:, 2], red_data[:, 1], bins=bins)
            ims.append(axes[row, 1].imshow(H, norm=LogNorm(), aspect='auto',
                                           extent=[min(bins[1]),max(bins[1]),min(bins[0]),max(bins[0])]))

            bins = [range(mp.array_size[0]), np.linspace(0, ap.sample_time * ap.numframes, 50)]
            H, _, _ = np.histogram2d(red_data[:, 2], red_data[:, 0], bins=bins)
            ims.append(axes[row, 2].imshow(H, norm=LogNorm(), aspect='auto',
                                           extent=[min(bins[1]),max(bins[1]),min(bins[0]),max(bins[0])]))

            bins = [np.linspace(0, ap.sample_time * ap.numframes, 50), np.linspace(-120, 0, 50)]
            H, _, _ = np.histogram2d(red_data[:, 0], red_data[:, 1], bins=bins)
            ims.append(axes[row, 3].imshow(H, norm=LogNorm(), aspect='auto',
                                           extent=[min(bins[1]),max(bins[1]),min(bins[0]),max(bins[0])]))

        fig.canvas.draw()
        # input("Press n to create a new figure or anything else to continue using this one")

    plt.show(block=True)

def show_performance(start=0, include_true_neg=False):
    """ Random assortment of prediction performance plots """
    alldata = load_meta()
    numsteps = len(alldata)

    # set axes
    plt.ion()
    plt.show(block=True)
    fig = plt.figure(figsize=(12,9), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4)

    true_ax = fig.add_subplot(gs[0, 0])
    guess_ax = fig.add_subplot(gs[0, 1])
    diff_ax = fig.add_subplot(gs[0, 2])

    xy_ax = fig.add_subplot(gs[1, 0])
    xt_ax = fig.add_subplot(gs[1, 1])
    xp_ax = fig.add_subplot(gs[1, 2])
    tp_ax = fig.add_subplot(gs[1, 3])

    xyim_ax = fig.add_subplot(gs[2, 0])
    xypos_ax = fig.add_subplot(gs[2, 1])
    xyneg_ax = fig.add_subplot(gs[2, 2])

    true_ax.set_title('True Label')
    guess_ax.set_title('Guess Label')
    diff_ax.set_title('Diff Label')
    true_ax.set_ylabel('Batch Num')

    for ax in [true_ax, guess_ax, diff_ax]:
        ax.set_xlabel('Point Num')

    for ax in [xy_ax, xt_ax, xp_ax, tp_ax]:
        ax.set_title('Guesses')

    xy_ax.set_xlabel('x')
    xy_ax.set_ylabel('y')
    xp_ax.set_xlabel('x')
    xp_ax.set_ylabel('phase')
    xt_ax.set_xlabel('x')
    xt_ax.set_ylabel('time')
    tp_ax.set_xlabel('time')
    tp_ax.set_ylabel('phase')

    xyim_ax.set_title('True all')
    xypos_ax.set_title('Planet Guess')
    xyneg_ax.set_title('Star Guess')

    for ax in [xy_ax, xyim_ax, xypos_ax, xyneg_ax]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()

    ims = []
    if start < 0:
        start = numsteps+start

    for epoch in range(start, numsteps):
        print(epoch)
        fig.suptitle(f'step {epoch}/{numsteps-1}', fontsize=16)

        if len(ims) > 0:
            for im in ims:
                im.remove()
            ims = []
        cur_seg, pred_seg_res, cur_data = alldata[epoch]

        # plot data
        ims.append(true_ax.imshow(cur_seg, aspect='auto'))
        ims.append(guess_ax.imshow(pred_seg_res, aspect='auto', vmax=1, vmin=0))
        ims.append(diff_ax.imshow(pred_seg_res - cur_seg, aspect='auto', vmax=1, vmin=-1))

        metrics = get_metrics(cur_seg, pred_seg_res, include_true_neg)
        colors = ['green', 'orange', 'purple', 'blue', ]
        alpha = 0.4
        for metric, c in zip(metrics, colors[:len(metrics)]):
            red_data = cur_data[metric]
            ims.append(xy_ax.scatter(red_data[:, 2], red_data[:, 3], c=c, marker='.', s=1, alpha=alpha))  # , marker=pids[0])
            ims.append(xp_ax.scatter(red_data[:, 2], red_data[:, 1], c=c, marker='.', s=1, alpha=alpha))  # , marker=pids[0])
            ims.append(xt_ax.scatter(red_data[:, 2], red_data[:, 0], c=c, marker='.', s=1, alpha=alpha))  # , marker=pids[0])
            ims.append(tp_ax.scatter(red_data[:, 0], red_data[:, 1], c=c, marker='.', s=1, alpha=alpha))  # , marker=pids[0])
        xy_ax.legend(['true_pos', 'false neg', 'false pos'])


        bins = [range(mp.array_size[0]), range(mp.array_size[1])]
        H, _, _ = np.histogram2d(cur_data[:, :, 3].flatten(), cur_data[:, :, 2].flatten(),
                                 bins=bins)
        ims.append(xyim_ax.imshow(H, norm=LogNorm()))

        positives = cur_data[pred_seg_res == 1]
        # print(pred_seg_res.shape, pred_seg_res[:10], pred_seg_res[:10] == 1, cur_data[:10], positives[:10])
        H, _, _ = np.histogram2d(positives[:, 3], positives[:, 2], bins=bins)
        ims.append(xypos_ax.imshow(H, norm=LogNorm()))

        negatives = cur_data[pred_seg_res == 0]
        H, _, _ = np.histogram2d(negatives[:, 3], negatives[:, 2], bins=bins)
        ims.append(xyneg_ax.imshow(H, norm=LogNorm()))

        fig.canvas.draw()
    
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    # show_performance()
    plot_metric_tesseracts()
