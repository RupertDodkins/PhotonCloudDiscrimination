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
from medis.Utils.misc import dprint
from pcd.config.medis_params import mp, ap
from pcd.config.config import config
import utils
import h5py

class Grid_Visualiser():
    def __init__(self, rows, cols, numsteps, row_headings=None, xtitles=None, ytitles=None):

        self.numsteps = numsteps

        # set axes
        plt.ion()
        plt.show(block=True)
        self.fig, self.axes = utils.init_grid(rows=rows, cols=cols, figsize=(12, 9))

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
            self.draw(self.steps[self.it], self.trainbools[self.it], self.images[self.it])

        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def update(self, step, trainbool, images, extent=None, norm=None):
        self.steps.append(step)
        self.trainbools.append(trainbool)
        self.images.append(images)

        self.it += 1

        self.draw(step, trainbool, images, extent, norm)

    def draw(self, step, trainbool, images, extent=None, norm=None):
        if not extent:
            extent = [[None for y in range(len(images[0])) ] for x in range(len(images))]

        epoch = step / (int(self.num_train / config['train']['batch_size']) + int(self.num_test / config['train']['batch_size']))

        kind = self.trainlabel[trainbool]
        print(step, kind, trainbool, epoch)
        self.fig.suptitle(f'Type: {kind}    |     step: {step}/{self.numsteps - 1}     |     epoch: {epoch:.2f}', fontsize=16)

        if len(self.ims) > 0:
            [im.remove() for im in self.ims]
            self.ims = []


        for ir in range(len(images)):
            for ic in range(len(images[0])):
                self.ims.append(self.axes[ir, ic].imshow(images[ir][ic], norm=norm, aspect='auto',
                                                         extent=extent[ir][ic]))  # ,
        # extent=[min(bins[1]),max(bins[1]),
        #        min(bins[0]),max(bins[0])]))

        self.fig.canvas.draw()

        # input("Press n to create a new figure or anything else to continue using this one")

    # plt.show(block=True)


def load_meta(kind='outputs'):
    alldata = []
    with open(config['train'][kind], 'rb') as handle:
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
    values = [np.empty(0)]*6

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
    metrics[kind]['Recall'] = np.append(metrics[kind]['Recall'], true_pos / (true_pos+false_neg))
    metrics[kind]['Precision'] = np.append(metrics[kind]['Precision'], true_pos / tot_pos)

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

    import tensorflow as tf
    import glob

    num_train, num_test = num_input()
    alldata = load_meta()
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)

    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall', 'Precision']
    axes = initialize_axes(metric_types+['Loss','Accuracy'])
    metrics = initialize_metrics(metric_types)

    for epoch in range(start, end+1):
        print(epoch)

        cur_seg, pred_seg_res, _, train = alldata[epoch]
        dprint(train)

        metrics = update_metrics(cur_seg, pred_seg_res, train, metrics)

    for kind in ['train', 'test']:
        # if kind == 'train':
        #     num_train = len(metrics[kind])#end+1 #- (end+1)//5
        #     epochs = np.arange(0, num_train, 0.5)
        # else:
        #     num_test = end // (1 / config['test_frac'])
        #     epochs = np.arange(0, num_test, 0.5)

        # get accuracies and losses
        dir = f"{config['working_dir']}/{kind}"
        lastfile = sorted(glob.glob(f'{dir}/*.thebeast'), key=os.path.getmtime)[-1]
        # lastfile = sorted(glob.glob(f'{dir}/*.thebeast'), key=os.path.getmtime)[-2]
        print(lastfile)
        losses, accuracies = [], []
        for e in tf.train.summary_iterator(lastfile):
            for v in e.summary.value:
                if v.tag == 'loss':
                    losses.append(v.simple_value)
                elif v.tag == 'accuracy':
                    accuracies.append(v.simple_value)

        # get the number of x values
        num = len(metrics[kind][metric_types[0]])
        if kind == 'train':
            epochs = np.linspace(0, num/int(num_train/config['train']['batch_size']), num)
        if kind == 'test':
            epochs = np.linspace(0, num/int(num_test/config['train']['batch_size']), num)

        print(kind, end+1, len(epochs), len(metrics[kind][metric_types[0]]))
        for ax, metric_key in zip(axes[:-2], metric_types):
            metrics[kind]['lines'].append(ax.plot(epochs, metrics[kind][metric_key], c=metrics[kind]['color'],
                                                  label=kind))

        axes[-2].plot(epochs, losses, c=metrics[kind]['color'], label=kind)
        axes[-1].plot(epochs, accuracies, c=metrics[kind]['color'], label=kind)
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
    try:
        conf = ('      +------+------+\n'
                '     1| %.2f | %.2f |\n'
                'Pred -+------+------+\n'
                '     0| %.2f | %.2f |\n'
                '      +------+------+\n'
                '         0   |  1\n'
                '           True' % (false_pos/tot_pos, true_pos/tot_pos,
                                     true_neg/tot_neg, false_neg/tot_neg))

        return conf
    except ZeroDivisionError:
        return ''


def continuous_metric_tesseracts():
    pass

# def visualise_grid(start, end, step, epoch_format):

def metric_tesseracts(start=-50, end=-1, jump=20):
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
    alldata = load_meta()
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)

    visualiser = Grid_Visualiser(4, 4, row_headings= ['True Planet','Missed Planet','True Star','Missed Star'],
                                 xtitles=['X','Phase','Time','Phase'], ytitles=['Y']*4, numsteps=allsteps)

    bins = [np.linspace(-2, 2, 50)] * 4
    dim_pairs = [[2, 3], [2, 1], [2, 0], [0, 1]]

    for step in range(start, end+1, jump):
        cur_seg, pred_seg_res, cur_data, trainbool = alldata[step]

        metrics = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)
        true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics[0])), int(np.sum(metrics[1])), \
                                                   int(np.sum(metrics[2])), int(np.sum(metrics[3])),

        print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg+false_neg, true_pos+false_pos))

        metrics[2], metrics[3] = metrics[3], metrics[2]

        images = [[]]*len(metrics)
        for row, metric in enumerate(metrics):
            red_data = cur_data[metric]

            images[row] = [[]]*len(dim_pairs)
            for ib, dim_pair in enumerate(dim_pairs):
                H, _, _ = np.histogram2d(red_data[:, dim_pair[0]], red_data[:, dim_pair[1]], bins=bins[ib])
                images[row][ib] = H

        # visualiser = Grid_Visualiser(4, 4, row_headings=['True Planet', 'Missed Planet', 'True Star', 'Missed Star'],
        #                              xtitles=['X', 'Phase', 'Time', 'Phase'], ytitles=['Y'] * 4, numsteps=allsteps)
        visualiser.update(step, trainbool, images, norm=LogNorm())
    plt.show(block=True)

def load_layers(start=-10, end=-1, jump=1, num_inputs=5, interactive=False):
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
    alldata = load_meta(kind='layers')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    num_layers = len(alldata[0][0])

    visualiser = Grid_Visualiser(rows=num_layers, cols=num_inputs, numsteps=allsteps,
                                 row_headings= ['Input','Layer 1','Layer 2','Layer 3','Layer 4'])

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
    alldata = load_meta(kind='layers')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    num_layers = len(alldata[0][0])
    print([pointcloud.shape[-1]==4 for pointcloud in alldata[0][0]])
    if not np.all([pointcloud.shape[-1]==4 for pointcloud in alldata[0][0]]):
        print('Skipping point clouds format display')
        return
    bins = np.linspace(-3,3,100)
    dim_pairs = [[2, 3], [2, 1], [2, 0], [0, 1]]
    num_inputs = len(dim_pairs)

    visualiser = Grid_Visualiser(rows=num_layers, cols=num_inputs, numsteps=allsteps,
                                 row_headings= ['Input','Layer 1','Layer 2','Layer 3','Layer 4'])

    for step in range(start, end + 1, jump):
        pointclouds, trainbool = alldata[step]

        images = [[]] * num_layers
        bounds = [[]] * num_layers
        for row in range(num_layers):
            pointcloud = pointclouds[row]

            images[row] = [[]] * num_inputs
            bounds[row] = [[]] * num_inputs
            for ic , dim_pair in enumerate(dim_pairs):
                H, b1, b2 = np.histogram2d(pointcloud[ib,:,dim_pair[0]],pointcloud[ib,:,dim_pair[1]])#, bins=50)#np.linspace(-3,3,100/(ic+1)))#, bins=bins)
                # ims.append(axes[row, ic].imshow(H, norm=LogNorm(), aspect='auto', extent=[min(b1),max(b1),min(b2),max(b2)]))
                extent=[min(b1),max(b1),min(b2),max(b2)]

                images[row][ic] = H
                bounds[row][ic] = extent
        visualiser.update(step, trainbool, images, bounds)
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    # onetime_metric_streams(end = args.epoch)
    # metric_tesseracts(start = 0, end = -1, jump=50)
    load_pointclouds(ib = 0, start = 0, end = -1, jump=50)
    load_layers(start = 0, end = -1, jump=50)
