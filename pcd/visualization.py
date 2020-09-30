"""
Monitors the predictions of a training model and plots various metrics

inputs
the pkl cache

#todo consolidate metric streams into a single class and do the same for metric tesseracts
"""

import os
import argparse
import pickle
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import numpy as np
import time
from medis.utils import dprint
from pcd.config.medis_params import mp, ap
from pcd.config.config import config
import utils
import h5py

class Grid_Visualiser():
    def __init__(self, rows, cols, numsteps, row_headings=None, xtitles=None, ytitles=None, norm=None):

        self.numsteps = numsteps

        # set axes
        plt.ion()
        plt.show(block=True)
        self.fig, self.axes = utils.init_grid(rows=rows, cols=cols, figsize=(12, 10))

        for i in range(rows):
            if row_headings:
                self.axes[i, 0].text(0.2, 0.2, row_headings[i], horizontalalignment='left',
                                     verticalalignment='center', transform=self.axes[i, 0].transAxes,
                                     fontsize=18)  #not fin
            if ytitles:
                self.axes[i, 0].set_ylabel(ytitles[i])

        if xtitles:
            for j in range(cols):
                self.axes[-1,j].set_xlabel(xtitles[j])

        self.ims = []
        self.steps = []
        self.trainbools = []
        self.images = []
        self.it = 0

        self.trainlabel = ['Test', 'Train']
        self.num_train, self.num_test = num_input()

        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                self.it -= 1
                print(self.it, event.button)
            else:
                self.it += 1
                print(self.it, event.button)
            self.draw(self.steps[self.it], self.trainbools[self.it], self.images[self.it], norm=norm)

        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def update(self, step, trainbool, images, extent=None, norm=None, vmax=None):
        self.steps.append(step)
        self.trainbools.append(trainbool)
        self.images.append(images)

        self.it += 1

        self.draw(step, trainbool, images, extent, norm, vmax)

    def draw(self, step, trainbool, images, extent=None, norm=None, vmax=None):
        # if not extent:
        #     extent = [[None for y in range(len(images[0])) ] for x in range(len(images))]
        if not vmax:
            vmax = [None]*len(images)

        epoch = step*config['train']['cache_freq'] / (int(self.num_train / config['train']['batch_size']) + int(self.num_test / config['train']['batch_size']))

        kind = self.trainlabel[trainbool]
        print(step, kind, trainbool, epoch)
        self.fig.suptitle(f'Type: {kind}    |     step: {step}/{self.numsteps - 1}     |     epoch: {epoch:.2f}', fontsize=16)

        self.fig.subplots_adjust(top=0.92, left=0.06, bottom=0.06)

        if len(self.ims) > 0:
            [im.remove() for im in self.ims]
            self.ims = []


        for ir in range(len(images)):
            for ic in range(len(images[0])):
                self.ims.append(self.axes[ir, ic].imshow(images[ir][ic], norm=norm, aspect='auto',
                                                         extent=extent, vmax=vmax[ir]))  # ,
        # extent=[min(bins[1]),max(bins[1]),
        #        min(bins[0]),max(bins[0])]))

        self.fig.canvas.draw()
        # input("Press n to create a new figure or anything else to continue using this one")

    # plt.show(block=True)


def load_meta(kind='outputs', amount=-1):
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

def get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True):
    true_neg = np.logical_and(cur_seg == 0, np.round(pred_seg_res) == 0)  # round just in case the pred_val is in mean mode
    true_pos = np.logical_and(cur_seg == 1, np.round(pred_seg_res) == 1)
    false_neg = np.logical_and(cur_seg == 1, np.round(pred_seg_res) == 0)
    false_pos = np.logical_and(cur_seg == 0, np.round(pred_seg_res) == 1)
    if include_true_neg:
        metrics = [true_pos, false_neg, false_pos, true_neg]
        # scores = np.array([np.sum(true_pos), np.sum(false_pos)]) / np.sum(cur_seg == 1)
        # print(scores)
        # scores = np.array([np.sum(true_neg), np.sum(false_neg)]) / np.sum(cur_seg == 0)
        # print(scores)
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
    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall', 'Precision']
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
    fig, axes = utils.init_grid(rows=2, cols=int(np.ceil(len(metric_types)/2)), figsize=(16, 9))
    axes = axes.flatten()
    for ax, metric_type in zip(axes, metric_types):
        ax.set_title(metric_type)
    return axes

def update_axes():
    pass

def initialize_metrics(metric_types):
    values = [np.empty(0)]*8

    metrics = {}
    metrics['train'] = dict(zip(metric_types, values))
    metrics['test'] = dict(zip(metric_types, values))
    print(metrics['train'])
    metrics['train']['lines'], metrics['test']['lines'] = [], []
    metrics['train']['color'], metrics['test']['color'] = 'C1', 'C0'

    return metrics

def update_metrics(cur_seg, pred_seg_res, train, metrics, loss=-1):
    metrics_vol = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)

    # np.float64 so ZeroDivideErrors -> np.nan
    true_pos, false_neg, false_pos, true_neg = np.float64(np.sum(metrics_vol[0])), np.float64(np.sum(metrics_vol[1])), \
                                               np.float64(np.sum(metrics_vol[2])), np.float64(np.sum(metrics_vol[3])),

    # tot_neg, tot_pos = true_neg + false_neg, true_pos + false_pos
    tot_neg, tot_pos = true_neg + false_pos, true_pos + false_neg
    dprint(tot_neg, tot_pos, train)

    kind = 'train' if train else 'test'
    metrics[kind]['True Positive'] = np.append(metrics[kind]['True Positive'], true_pos / tot_pos)
    metrics[kind]['False Negative'] = np.append(metrics[kind]['False Negative'], false_neg / tot_pos)
    metrics[kind]['False Positive'] = np.append(metrics[kind]['False Positive'], false_pos / tot_neg)
    metrics[kind]['True Negative'] = np.append(metrics[kind]['True Negative'], true_neg / tot_neg)
    metrics[kind]['Recall'] = np.append(metrics[kind]['Recall'], true_pos / (true_pos+false_neg))
    metrics[kind]['Precision'] = np.append(metrics[kind]['Precision'], true_pos / (true_pos + false_pos))
    metrics[kind]['Accuracy'] = np.append(metrics[kind]['Accuracy'], (true_pos+true_neg) / (tot_pos+tot_neg))
    metrics[kind]['Loss'] = np.append(metrics[kind]['Loss'], loss)

    return metrics

def num_input():
    ALL_FILES = np.append(config['trainfiles'], config['testfiles'])
    train_inds = 0
    test_inds = 0
    for t, file in enumerate(ALL_FILES):
        with h5py.File(file, 'r') as hf:
            if 'train' in file:
                train_inds += len(hf.get('data'))
            else:
                test_inds += len(hf.get('data'))
    return train_inds, test_inds

def onetime_metric_streams(start=0, end=10):
    """
    Shows metrics as a function of training steps

    :return:
    """

    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()
    # import glob

    num_train, num_test = num_input()
    alldata = load_meta('pt_outputs')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)

    # metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall', 'Precision']
    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall', 'Precision',
                    'Accuracy', 'Loss']
    # axes = initialize_axes(metric_types+['Loss','Accuracy'])
    axes = initialize_axes(metric_types)
    metrics = initialize_metrics(metric_types)

    for step in range(start, end+1):
        dprint(step)

        cur_seg, pred_seg_res, _, loss, train = alldata[step]
        dprint(train)

        metrics = update_metrics(cur_seg, pred_seg_res, train, metrics, loss)

    for kind in ['train', 'test']:
        # if kind == 'train':
        #     num_train = len(metrics[kind])#end+1 #- (end+1)//5
        #     epochs = np.arange(0, num_train, 0.5)
        # else:
        #     num_test = end // (1 / config['test_frac'])
        #     epochs = np.arange(0, num_test, 0.5)

        # get accuracies and losses
        # dir = f"{config['working_dir']}/{kind}"
        # # lastfile = sorted(glob.glob(f'{dir}/*.thebeast'), key=os.path.getmtime)[-1]
        # lastfile = sorted(glob.glob(f'{dir}/*.glados*'), key=os.path.getmtime)[-1]
        # print(dir, lastfile)
        # # lastfile = sorted(glob.glob(f'{dir}/*.thebeast'), key=os.path.getmtime)[-2]
        # print(lastfile)
        # losses, accuracies = [], []
        # from tensorflow.python.summary.summary_iterator import summary_iterator
        # for e in tf.train.summary_iterator(lastfile):
        #     for v in e.summary.value:
        #         if v.tag == 'loss' or v.tag == 'batch_loss':
        #             losses.append(v.simple_value)
        #         elif v.tag == 'accuracy' or v.tag == 'batch_sparse_categorical_accuracy':
        #             accuracies.append(v.simple_value)


        # get the number of x values
        num = len(metrics[kind][metric_types[0]])
        if kind == 'train':
            epochs = np.linspace(0, num * config['train']['cache_freq']/int(num_train/config['train']['batch_size']), num)
        if kind == 'test':
            epochs = np.linspace(0, num * config['train']['cache_freq']/int(num_test/config['train']['batch_size']), num)

        # dprint(kind, end+1, len(epochs), len(metrics[kind][metric_types[0]]), losses)
        for ax, metric_key in zip(axes, metric_types):
            metrics[kind]['lines'].append(ax.plot(epochs, metrics[kind][metric_key], c=metrics[kind]['color'],
                                                  label=kind))

        # # sometimes losses
        # try:
        #     axes[-2].plot(epochs, losses[:len(epochs)], c=metrics[kind]['color'], label=kind)
        #     axes[-1].plot(epochs, accuracies[:len(epochs)], c=metrics[kind]['color'], label=kind)
        # except ValueError:
        #     print(f'epochs and losses wrong length ?! {len(epochs)}, {len(losses)} respectively')
    axes[2].legend()

    plt.show(block=True)

def plot_summary_data():
    """ Functionality should have been incorporated into onetime metric streams """
    import tensorflow as tf
    import glob

    num_train, num_test = num_input()
    fig, axes = utils.init_grid(rows=1, cols=2, figsize=(12, 9))

    for kind in ['train', 'test']:
        dir = f"{config['working_dir']}/{kind}"
        lastfile = sorted(glob.glob(f'{dir}/*.thebeast'), key=os.path.getmtime)[-1]
        print(lastfile)
        losses, accuracies = [], []
        for e in tf.train.summary_iterator(lastfile):
            for v in e.summary.value:
                if v.tag == 'loss':
                    losses.append(v.simple_value)
                elif v.tag == 'accuracy':
                    accuracies.append(v.simple_value)

        num = len(losses)
        if kind == 'train':
            epochs = np.linspace(0, num / int(num_train / config['train']['batch_size']), num)
        if kind == 'test':
            epochs = np.linspace(0, num / int(num_test / config['train']['batch_size']), num)

        axes[0, 0].plot(epochs, losses, label=kind)
        axes[0, 1].plot(epochs, accuracies, label=kind)

    plt.show()

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
    return conf



# def continuous_metric_tesseracts():
#     pass
#
# # def visualise_grid(start, end, step, epoch_format):

def check_inputs(ind=None):
    alldata = load_meta()
    allsteps = len(alldata)

    # fig, axes = utils.init_grid(rows=config['classes'], cols=config['dimensions'])
    # # fig, axes = utils.init_grid(rows=self.num_classes, cols=4)
    # fig.suptitle(f'{ind}', fontsize=16)
    # plt.tight_layout()
    #
    # bins = [np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.linspace(-1, 1, 150), np.linspace(-1, 1, 150)]
    #
    # # coord = 'tpxy'
    # coord = 'pytx'
    #
    # for o in range(config['classes']):
    #     if ind:
    #         H, _ = np.histogramdd(self.data[ind, (self.labels[ind] == o)], bins=bins)
    #     else:
    #         H, _ = np.histogramdd(self.data[self.labels == o], bins=bins)
    #
    #     for p, pair in enumerate([['x', 'y'], ['x', 'p'], ['x', 't'], ['p', 't']]):
    #         inds = coord.find(pair[0]), coord.find(pair[1])
    #         sumaxis = tuple(np.delete(range(len(coord)), inds))
    #         image = np.sum(H, axis=sumaxis)
    #         if pair in [['x', 'p'], ['x', 't']]:
    #             image = image.T
    #             inds = inds[1], inds[0]
    #         axes[o, p].imshow(image, norm=LogNorm(), aspect='auto',
    #                           extent=[bins[inds[0]][0], bins[inds[0]][-1], bins[inds[1]][0], bins[inds[1]][-1]])
    #
    # plt.show(block=True)

def metric_tesseracts(start=-50, end=-1, jump=1, type='both'):
    """ Shows the net predictions on the cloud as a series of 2d histograms in 4x4 grid of form
     ___________________________
    |_______|
    |_layer_|
    |___TP__|
    |___FP__|
    |___TN__|
    |___FN__|

    """

    assert end != 0
    # assert jump >= config['train']['cache_freq']
    alldata = load_meta('pt_outputs')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    dprint(start, end)

    visualiser = Grid_Visualiser(4, 4, row_headings= ['True Planet','Missed Planet','True Star','Missed Star'],
                                 xtitles=['X','Phase','Time','Phase'], ytitles=['Y']*4, numsteps=allsteps,
                                 norm=LogNorm())

    if config['data']['quantize']:
        _,_, cur_data, _, _ = alldata[0]
        bins = [np.linspace(np.min(cur_data[:,0]),np.max(cur_data[:,0]), 100)] * 4
    else:
        bins = [np.linspace(-1, 1, 100) * 1e6] * 4

    # dim_pairs = [[2, 3], [2, 1], [2, 0], [0, 1]]
    # dim_pairs = [[3, 1], [3, 0], [3, 2], [2, 0]]
    # dim_pairs = [[3, 1], [3, 2], [3, 0], [1, 2]]
    dim_pairs = [[3, 2], [3, 1], [3, 0], [1, 2]]
    # dim_pairs = np.array(dim_pairs)[[1, 3, 0, 2]]

    for step in range(start, end+1, jump):
        cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[step]

        metrics = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)
        true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics[0])), int(np.sum(metrics[1])), \
                                                   int(np.sum(metrics[2])), int(np.sum(metrics[3])),

        print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))
        print('true_pos: %f' % (true_pos))
        print('true_neg: %f' % (true_neg))
        print('false_pos: %f' % (false_pos))
        print('false_neg: %f' % (false_neg))
        try:  # sometimes there are no true_positives
            print('Recall: %f' % (true_pos / (true_pos + false_neg)))
            print('Precision: %f' % (true_pos / (true_pos + false_pos)))
        except ZeroDivisionError:
            pass

        metrics[2], metrics[3] = metrics[3], metrics[2]

        images = [[]]*len(metrics)
        for row, metric in enumerate(metrics):
            red_data = cur_data[metric]
            # if config['data']['trans_polar']:
            #     red_data = trans_p2c(red_data)

            images[row] = [[]]*len(dim_pairs)
            for ib, dim_pair in enumerate(dim_pairs):
                H, _, _ = np.histogram2d(red_data[:, dim_pair[0]], red_data[:, dim_pair[1]], bins=bins[ib])
                images[row][ib] = H

        # visualiser = Grid_Visualiser(4, 4, row_headings=['True Planet', 'Missed Planet', 'True Star', 'Missed Star'],
        #                              xtitles=['X', 'Phase', 'Time', 'Phase'], ytitles=['Y'] * 4, numsteps=allsteps)
        visualiser.update(step, trainbool, images, norm=None, extent=[-1,1,-1,1])
    plt.show(block=True)

def investigate_layer(start=-10, end=-1, jump=1):
    """ Shows the net layers of the cloud as a series of 2d histograms in num_inputsx4 grid of form
     __________________________________
    |_____________|____batch_index____|
    |____layer____|_0_|_1_|_2_|...|_n_|
    |__pred_star__|___|___|___|...|___|
    |__pred_comp__|___|___|___|...|___|
    |_total_layer_|___|___|___|...|___|
    |__pred_star__|___|___|___|...|___|
    |__pred_comp__|___|___|___|...|___|

    """

    assert end != 0
    # assert jump >= config['train']['cache_freq']

    all_layers = load_meta(kind='layers')
    all_outputs = load_meta(kind='outputs')

    allsteps = len(all_layers)
    start, end = get_range_inds(start, end, allsteps)

    visualiser = Grid_Visualiser(rows=6, cols=config['train']['batch_size'], numsteps=allsteps,
                                 row_headings= ['pred_star', 'pred_comp', 'correct planet', 'missed planet',
                                                'missed star', 'correct star']
                                 )
    bins = np.linspace(-1, 1, 150)

    for l in range(9):
        for step in range(start, end + 1, jump):
            layer, trainbool = all_layers[step]
            layer = layer[l]
            batch_labels, pred_val, batch_data, _ = all_outputs[step]

            # images = [[[]] * config['train']['batch_size']]*5

            images = [[]] * 6
            for i in range(6):
                images[i] = [[]] * config['train']['batch_size']

            for ic in range(config['train']['batch_size']):
                x, y = batch_data[ic, :, :2][~(np.ones(2) * pred_val[ic, :, None]).astype(bool)].reshape(-1,2).T
                images[0][ic], _, _ = np.histogram2d(x,y, bins=bins)

                x, y = batch_data[ic, :, :2][(np.ones(2) * pred_val[ic, :, None]).astype(bool)].reshape(-1,2).T
                images[1][ic], _, _ = np.histogram2d(x,y, bins=bins)

                metrics = get_metric_distributions(batch_labels[ic], pred_val[ic], include_true_neg=True)

                if layer.shape[-1] != 4:
                    for ii, im in enumerate(range(2,6)):
                        images[im][ic] = layer[ic] * (np.ones(layer.shape[-1])*metrics[ii][:,None])
                else:
                    for ii, im in enumerate(range(2, 6)):
                        x, y = layer[ic,:,:2][(np.ones(2)*metrics[ii][:,None]).astype(bool)].reshape(-1,2).T
                        images[im][ic] = np.histogram2d(x, y, bins=150)[0]

            visualiser.update(step, trainbool, images)#, vmax=[None, None, 1, 1, 1, 1])
    plt.show(block=True)

def load_layers(start=-10, end=-1, jump=1, num_inputs=3, interactive=False):
    """ Shows the net layers of the cloud as a series of 2d histograms in num_inputsx4 grid of form
     ___________________________
    |_______|____batch_index____|
    |_layer_|_0_|_1_|_2_|...|_n_|
    |_input_|___|___|___|...|___|
    |___1___|___|___|___|...|___|
    |___2___|___|___|___|...|___|
    |___3___|___|___|___|...|___|
    |___4___|___|___|___|...|___|
    |   .   | . | . | . |   | . |
    |   .   | . | . | . |   | . |
    |___.___|_._|_._|_._|   |_._|
    |___m___|___|___|___|...|___|
    """

    assert num_inputs <= config['train']['batch_size']
    assert end != 0
    # assert jump >= config['train']['cache_freq']

    alldata = load_meta(kind='layers')

    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    num_layers = len(alldata[0][0])

    visualiser = Grid_Visualiser(rows=num_layers, cols=num_inputs, numsteps=allsteps,)
                                 #row_headings= ['Input','Layer 1','Layer 2','Layer 3','Layer 4'])

    for step in range(start, end + 1, jump):
        pointclouds, trainbool = alldata[step]

        images = [[]] * num_layers
        for ir, row in enumerate(range(num_layers)):
            pointcloud = pointclouds[row]

            images[row] = [[]] * num_inputs
            for ic in range(num_inputs):
                images[row][ic] = pointcloud[ic]

        # if not interactive:
        #     visualiser = Grid_Visualiser(rows=num_layers, cols=num_inputs, numsteps=allsteps,
        #                                  row_headings=['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
        visualiser.update(step, trainbool, images)
    plt.show(block=True)

def load_pointclouds(start=-10, end=-1, jump=1, ib=0):
    """ Shows the net layers of the cloud as a series of 2d histograms in num_inputsx4 grid of form
     ___________________________
    |_______|_______Pair________|
    |_layer_|_xy_|_py_|_ty_|_pt_|
    |_input_|____|____|____|____|
    |___1___|____|____|____|____|
    |___2___|____|____|____|____|
    |   .   |  . |  . |  . |  . |
    |   .   |  . |  . |  . |  . |
    |___.___|__._|__._|__._|__._|
    |___m___|____|____|____|____|


    """

    assert end != 0
    # assert jump >= config['train']['cache_freq']

    alldata = load_meta(kind='layers')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    # if np.array([len(pc.shape)==3 for pc in alldata[0][0]]).all():
    num_layers = len(alldata[0][0])
    # else:

    print([pointcloud.shape[-1]==4 for pointcloud in alldata[0][0]])
    if not np.all([pointcloud.shape[-1]==4 for pointcloud in alldata[0][0]]):
        print('Skipping point clouds format display')
        return
    # bins = np.linspace(-3,3,100)
    # bins = [np.linspace(-1, 1, 150)] * 4
    bins = [150]*4
    # dim_pairs = [[2, 3], [2, 1], [2, 0], [0, 1]]
    dim_pairs = [[3, 1], [3, 2], [3, 0], [1, 2]]
    num_inputs = len(dim_pairs)

    visualiser = Grid_Visualiser(rows=num_layers, cols=num_inputs, numsteps=allsteps,
                                 # row_headings= ['Input','Aug','Gather','Group0','Group1','Group2'])
                                row_headings=None
                                 )

    for step in range(start, end + 1, jump):
        pointclouds, trainbool = alldata[step]

        images = [[]] * num_layers
        bounds = [[]] * num_layers
        for row in range(num_layers):
            pointcloud = pointclouds[row]

            images[row] = [[]] * num_inputs
            bounds[row] = [[]] * num_inputs
            for ic , dim_pair in enumerate(dim_pairs):
                H, b1, b2 = np.histogram2d(pointcloud[ib,:,dim_pair[0]],pointcloud[ib,:,dim_pair[1]], bins[ic])#, bins=50)#np.linspace(-3,3,100/(ic+1)))#, bins=bins)
                # ims.append(axes[row, ic].imshow(H, norm=LogNorm(), aspect='auto', extent=[min(b1),max(b1),min(b2),max(b2)]))
                extent=[min(b1),max(b1),min(b2),max(b2)]

                images[row][ic] = H
                bounds[row][ic] = extent
        visualiser.update(step, trainbool, images, extent=[-1,1,-1,1], norm=LogNorm())
    plt.show(block=True)

def trans_p2c(photons):
    # photons[2] += 1
    # photons[2] *= mp.array_size[1]/2
    photons[:, 3] *= np.pi
    photons[:, 2] += 1

    x = photons[:, 2] * np.cos(photons[:, 3])
    y = photons[:, 2] * np.sin(photons[:, 3])

    # x += mp.array_size[1]/2
    # y += mp.array_size[0]/2

    photons[:, 2], photons[:, 3] = x, y

    return photons

def pt_step(input_data, input_label, pred_val, loss, train=True):
    pred_val = np.argmax(pred_val, axis=-1)
    with open(config['train']['pt_outputs'], 'ab') as handle:
        field_tup = (input_label, pred_val, input_data, loss, train)
        pickle.dump(field_tup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pos = input_label == 1
    neg = input_label == 0

    true_pos = int(np.sum(np.logical_and(pos, np.round(pred_val) == 1)))
    false_pos = int(np.sum(np.logical_and(neg, np.round(pred_val) == 1)))
    true_neg = int(np.sum(np.logical_and(neg, np.round(pred_val) == 0)))
    false_neg = int(np.sum(np.logical_and(pos, np.round(pred_val) == 0)))
    conf = confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg)

    print('true_pos: %f' % (true_pos))
    print('true_neg: %f' % (true_neg))
    print('false_pos: %f' % (false_pos))
    print('false_neg: %f' % (false_neg))
    print(conf)
    try:
        print('Precision: %f' % (true_pos / (true_pos + false_pos)))
        print('Recall: %f' % (true_pos / (true_pos + false_neg)))
    except ZeroDivisionError:
        pass

def tf_step(input_data, input_label, pred_val, train=True):
    """ Get values of tensors to save them and read by metric_tesseracts """

    if not isinstance(pred_val, np.ndarray):
        pred_val = pred_val.numpy()
    if not isinstance(input_label, np.ndarray):
        input_label = input_label.numpy()
    if not isinstance(input_data, np.ndarray):
        input_data = input_data.numpy()
    if not isinstance(train, bool):
        train = train.numpy()

    if len(pred_val.shape) > 1:
        pred_val = np.argmax(pred_val, axis=-1)

    with open(config['train']['outputs'], 'ab') as handle:
        field_tup = (input_label, pred_val, input_data, train)
        pickle.dump(field_tup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pos = input_label == 1
    neg = input_label == 0

    true_pos = int(np.sum(np.logical_and(pos, np.round(pred_val) == 1)))
    false_pos = int(np.sum(np.logical_and(neg, np.round(pred_val) == 1)))
    true_neg = int(np.sum(np.logical_and(neg, np.round(pred_val) == 0)))
    false_neg = int(np.sum(np.logical_and(pos, np.round(pred_val) == 0)))
    conf = confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos,
                            true_pos + false_neg)

    print('true_pos: %f' % (true_pos))
    print('true_neg: %f' % (true_neg))
    print('false_pos: %f' % (false_pos))
    print('false_neg: %f' % (false_neg))
    print(conf)
    try:
        print('Precision: %f' % (true_pos / (true_pos + false_pos)))
        print('Recall: %f' % (true_pos / (true_pos + false_neg)))
    except ZeroDivisionError:
        pass

    return 1, 1, 1, True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    # onetime_metric_streams(end = -1)
    metric_tesseracts(start = 0, end = -1, jump=1)

