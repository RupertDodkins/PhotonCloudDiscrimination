"""
Monitors the predictions of a training model and plots various metrics

inputs
the pkl cache
"""

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

def get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True):
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
    metrics = get_metric_distributions(cur_seg, pred_seg_res)
    three_d_scatter(cur_data, metrics)

def metric_streams(start=0, end=10):
    """
    Shows metrics as a function of training steps

    :return:
    """

    alldata = load_meta()
    allsteps = len(alldata)
    if start < 0:
        start = allsteps+start
    if end < 0:
        end = allsteps + end

    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Precision', 'Recall']
    fig, axes = utils.init_grid(rows=2, cols=len(metric_types), figsize=(12,9))
    axes = axes.flatten()

    true_pos_stream, false_neg_stream, false_pos_stream, true_neg_stream = [], [], [], []
    for epoch in range(start, end+1):
        print(epoch)

        cur_seg, pred_seg_res, cur_data = alldata[epoch]

        metrics_vol = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)
        true_pos_stream.append(int(np.sum(metrics_vol[0])))
        false_neg_stream.append(int(np.sum(metrics_vol[1])))
        false_pos_stream.append(int(np.sum(metrics_vol[2])))
        true_neg_stream.append(int(np.sum(metrics_vol[3])))

    true_pos_stream = np.array(true_pos_stream)
    false_neg_stream = np.array(false_neg_stream)
    false_pos_stream = np.array(false_pos_stream)
    true_neg_stream = np.array(true_neg_stream)
    precision = true_pos_stream/(true_pos_stream+false_pos_stream)
    recall = true_pos_stream/(true_pos_stream+false_neg_stream)

    metrics = [true_pos_stream, false_neg_stream, false_pos_stream, true_neg_stream, precision, recall]
    for ax, metric_type, metric in zip(axes, metric_types, metrics):
        ax.plot(metric)
        ax.set_title(metric_type)

    plt.show(block=True)

def confusion_matrix(false_neg, true_pos, true_neg, false_pos, tot_neg, tot_pos):
    return ('      +------+------+\n'
            '     1| %.2f | %.2f |\n'
            'Pred -+------+------+\n'
            '     0| %.2f | %.2f |\n'
            '      +------+------+\n'
            '         0   |  1\n'
            '           True' % (false_pos/tot_pos, true_pos/tot_pos,
                                 true_neg/tot_neg, false_neg/tot_neg))

def metric_tesseracts(start=0, end=-1, include_true_neg=True):
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

        metrics = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    metric_tesseracts(end = args.epoch)
    metric_streams(end = args.epoch)