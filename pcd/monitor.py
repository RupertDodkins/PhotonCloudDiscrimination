import os
import argparse
import pickle
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import numpy as np
from config.medis_params import mp
from config.config import config

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

def show_performance(epoch=-1, include_true_neg=False):
    alldata = load_meta()

    # print(alldata.shape, alldata.T.shape)
    # for ground, pred in zip(cur_seg[num_epoch-1:], pred_seg_res[num_epoch-1:]):
    cur_seg, pred_seg_res, cur_data = alldata[epoch]
    print(cur_data.shape)

    # set axes
    fig3 = plt.figure(figsize=(9,9), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3)

    true_ax = fig3.add_subplot(gs[0, 0])
    guess_ax = fig3.add_subplot(gs[0, 1])
    diff_ax = fig3.add_subplot(gs[0, 2])

    xy_ax = fig3.add_subplot(gs[1, 0])
    xp_ax = fig3.add_subplot(gs[1, 1])
    xt_ax = fig3.add_subplot(gs[1, 2])
    tp_ax = fig3.add_subplot(gs[2, 0])

    xyim_ax = fig3.add_subplot(gs[2, 1])
    xypos_ax = fig3.add_subplot(gs[2, 2])

    true_ax.set_title('True')
    guess_ax.set_title('Guess')
    diff_ax.set_title('Diff')
    true_ax.set_ylabel('Label')

    for ax in [true_ax, guess_ax, diff_ax]:
        ax.set_xlabel('Batch Input')

    for ax in [xy_ax, xp_ax, xt_ax, tp_ax]:
        ax.set_title('Guesses')

    xy_ax.set_label(('x','y'))
    xp_ax.set_label(('x','p'))
    xt_ax.set_label(('x','t'))
    tp_ax.set_label(('t','p'))

    xyim_ax.set_title('True all')
    xypos_ax.set_title('Planet Guess')

    # plot data
    true_ax.imshow(cur_seg, aspect='auto')
    guess_ax.imshow(pred_seg_res, aspect='auto')
    diff_ax.imshow(pred_seg_res - cur_seg, aspect='auto')

    metrics = get_metrics(cur_seg, pred_seg_res, include_true_neg)
    colors = ['green', 'orange', 'purple', 'blue', ]
    for metric, c in zip(metrics, colors[:len(metrics)]):
        red_data = cur_data[metric]
        xy_ax.scatter(red_data[:, 2], red_data[:, 3], c=c, marker='.', s=1)  # , marker=pids[0])
        xp_ax.scatter(red_data[:, 2], red_data[:, 0], c=c, marker='.', s=1)  # , marker=pids[0])
        xt_ax.scatter(red_data[:, 2], red_data[:, 1], c=c, marker='.', s=1)  # , marker=pids[0])
        tp_ax.scatter(red_data[:, 0], red_data[:, 1], c=c, marker='.', s=1)  # , marker=pids[0])


    bins = [range(mp.array_size[0]), range(mp.array_size[1])]
    H, _, _ = np.histogram2d(cur_data[:, :, 3].flatten(), cur_data[:, :, 2].flatten(),
                             bins=bins)
    xyim_ax.imshow(H, norm=LogNorm())

    positives = cur_data[pred_seg_res == 1]
    print(pred_seg_res.shape, pred_seg_res[:10], pred_seg_res[:10] == 1, cur_data[:10], positives[:10])
    H, _, _ = np.histogram2d(positives[:, 2], positives[:, 3], bins=bins)
    xypos_ax.imshow(H, norm=LogNorm())

    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    show_performance(38)
