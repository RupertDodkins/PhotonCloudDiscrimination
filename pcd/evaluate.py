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
from medis.Utils.misc import dprint
from config.medis_params import mp, ap
from config.config import config
import utils

def load_meta():
    alldata = []
    with open(config['train']['outputs'], 'rb') as handle:
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
    cur_seg, pred_seg_res, cur_data, _ = alldata[epoch]
    metrics = get_metric_distributions(cur_seg, pred_seg_res)
    three_d_scatter(cur_data, metrics)

def continuous_metric_streams():
    while not os.path.exists(config['train']['outputs']):
        print('Waiting for model to output first metric data')
        time.sleep(5)

    # set axes
    plt.ion()
    plt.show(block=True)
    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall']
    # fig, axes = utils.init_grid(rows=2, cols=int(np.ceil(len(metric_types)/2)), figsize=(12, 9))
    # axes = axes.flatten()
    # for ax, metric_type in zip(axes, metric_types):
    #     ax.set_title(metric_type)
    #
    # metric_keys = ['true_pos_stream', 'false_neg_stream', 'false_pos_stream', 'true_neg_stream', 'recall']
    # values = [np.empty(0)]*5
    #
    # metrics = {}
    # metrics['train'] = dict(zip(metric_keys, values))
    # metrics['test'] = dict(zip(metric_keys, values))
    # print(metrics['train'])
    # metrics['train']['lines'], metrics['test']['lines'] = [], []
    # metrics['train']['color'], metrics['test']['color'] = 'C1', 'C0'

    axes = initialize_axes(metric_types)
    metrics = initialize_metrics(metric_types)

    epoch = 0
    while True:
        alldata = load_meta()
        if len(alldata) > 1 and len(metrics['train']['true_pos_stream'])==0:
            print('*** Warning *** ML cache already has mutliple time steps saved. Consider running metric_streams or '
                  'creating new file. Stopping continuous_streams.')
            # break

        dprint(len(alldata), len(metrics['train']['true_pos_stream']), len(metrics['train']['true_pos_stream']))

        if len(alldata) == len(metrics['train']['true_pos_stream'])+len(metrics['train']['true_pos_stream']) + 1:
            cur_seg, pred_seg_res, cur_data, train = alldata[-1]  # will get progressively slower as whole file load is required to get final element

            if train:
                # todo unhard code this
                epoch += 0.25
                epochs = np.arange(0,epoch,0.25)
            else:
                epochs = np.arange(0, epoch, 1)

            # metrics_vol = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)
            #
            # true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics_vol[0])), int(np.sum(metrics_vol[1])), \
            #                                            int(np.sum(metrics_vol[2])), int(np.sum(metrics_vol[3])),
            #
            # tot_neg, tot_pos = true_neg + false_neg, true_pos + false_pos
            # dprint(tot_neg, tot_pos, train)
            #
            kind = 'train' if train else 'test'
            # metrics[kind]['true_pos_stream'] = np.append(metrics[kind]['true_pos_stream'], true_pos/tot_pos)
            # metrics[kind]['false_neg_stream'] = np.append(metrics[kind]['false_neg_stream'], false_neg/tot_neg)
            # metrics[kind]['false_pos_stream'] = np.append(metrics[kind]['false_pos_stream'], false_pos/tot_pos)
            # metrics[kind]['true_neg_stream'] = np.append(metrics[kind]['true_neg_stream'], true_neg/tot_neg)
            #
            # if metrics[kind]['true_pos_stream'][-1] + metrics[kind]['false_neg_stream'][-1] == 0:
            #     metrics[kind]['recall'] = np.append(metrics[kind]['recall'], np.nan)
            # else:
            #     metrics[kind]['recall'] = np.append(metrics[kind]['recall'],
            #                                         metrics[kind]['true_pos_stream'][-1] /
            #                                         (metrics[kind]['true_pos_stream'][-1] +
            #                                          metrics[kind]['false_neg_stream'][-1]))
            metrics = update_metrics(cur_seg, pred_seg_res, kind, metrics)

            if len(metrics[kind]['lines']) > 0:
                [line.pop(0).remove() for line in metrics[kind]['lines']]
                metrics[kind]['lines'] = []

            dprint(len(epochs), len(metrics[kind][metric_types[0]]), epochs, metrics[kind][metric_types[0]])
            for ax, metric_key in zip(axes, metric_types):
                metrics[kind]['lines'].append(ax.plot(epochs, metrics[kind][metric_key], c=metrics['train']['color']))

            fig.canvas.draw()
        else:
            print('No new data yet')
            time.sleep(10)
            # continue
        time.sleep(0.01)

def initialize_axes(metric_types):
    fig, axes = utils.init_grid(rows=2, cols=int(np.ceil(len(metric_types)/2)), figsize=(12, 9))
    axes = axes.flatten()
    for ax, metric_type in zip(axes, metric_types):
        ax.set_title(metric_type)
    return axes

def update_axes():
    pass

def initialize_metrics(metric_types):
    values = [np.empty(0)]*5

    metrics = {}
    metrics['train'] = dict(zip(metric_types, values))
    metrics['test'] = dict(zip(metric_types, values))
    print(metrics['train'])
    metrics['train']['lines'], metrics['test']['lines'] = [], []
    metrics['train']['color'], metrics['test']['color'] = 'C1', 'C0'

    return metrics

def update_metrics(cur_seg, pred_seg_res, train, metrics):
    metrics_vol = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)

    # np.float64 so ZeroDivideErrors -> np.nan
    true_pos, false_neg, false_pos, true_neg = np.float64(np.sum(metrics_vol[0])), np.float64(np.sum(metrics_vol[1])), \
                                               np.float64(np.sum(metrics_vol[2])), np.float64(np.sum(metrics_vol[3])),

    tot_neg, tot_pos = true_neg + false_neg, true_pos + false_pos
    dprint(tot_neg, tot_pos, train)

    kind = 'train' if train else 'test'
    metrics[kind]['True Positive'] = np.append(metrics[kind]['True Positive'], true_pos / tot_pos)
    metrics[kind]['False Negative'] = np.append(metrics[kind]['False Negative'], false_neg / tot_neg)
    metrics[kind]['False Positive'] = np.append(metrics[kind]['False Positive'], false_pos / tot_pos)
    metrics[kind]['True Negative'] = np.append(metrics[kind]['True Negative'], true_neg / tot_neg)
    metrics[kind]['Recall'] = np.append(metrics[kind]['Recall'], true_pos / (true_pos/false_neg))

    return metrics


def onetime_metric_streams(start=0, end=10):
    """
    Shows metrics as a function of training steps

    :return:
    """

    alldata = load_meta()
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)

    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall']
    axes = initialize_axes(metric_types)
    metrics = initialize_metrics(metric_types)

    for epoch in range(start, end+1):
        print(epoch)

        cur_seg, pred_seg_res, _, train = alldata[epoch]
        dprint(train)

        metrics = update_metrics(cur_seg, pred_seg_res, train, metrics)

    for kind in ['train', 'test']:
        if kind == 'train':
            num_train = end+1 - (end+1)//5
            epochs = np.arange(0, num_train/4, 0.25)
        else:
            num_test = end // 5
            epochs = np.arange(0, num_test, 1)

        print(kind, end+1, len(epochs), len(metrics[kind][metric_types[0]]))
        for ax, metric_key in zip(axes, metric_types):
            metrics[kind]['lines'].append(ax.plot(epochs, metrics[kind][metric_key], c=metrics[kind]['color'],
                                                  label=kind))
    axes[2].legend()

    plt.show(block=True)

def get_range_inds(start, end, allsteps):
    if start < 0:
        start = allsteps+start
    if end < 0:
        end = allsteps + end
    return start, end

def confusion_matrix(false_neg, true_pos, true_neg, false_pos, tot_neg, tot_pos):
    return ('      +------+------+\n'
            '     1| %.2f | %.2f |\n'
            'Pred -+------+------+\n'
            '     0| %.2f | %.2f |\n'
            '      +------+------+\n'
            '         0   |  1\n'
            '           True' % (false_pos/tot_pos, true_pos/tot_pos,
                                 true_neg/tot_neg, false_neg/tot_neg))

def continuous_metric_tesseracts():
    pass

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
    start, end = get_range_inds(start, end, allsteps)

    time.sleep(5)

    for epoch in range(start, end+1):
        print(epoch)
        fig.suptitle(f'step {epoch}/{allsteps-1}', fontsize=16)

        if len(ims) > 0:
            [im.remove() for im in ims]
            ims = []

        cur_seg, pred_seg_res, cur_data, _ = alldata[epoch]

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
    onetime_metric_streams(end = args.epoch)
    metric_tesseracts(end = args.epoch)