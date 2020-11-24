"""
Monitors the predictions of a training model and plots various metrics

inputs
the pkl cache

"""

import argparse
import pickle
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import numpy as np
from medis.utils import dprint
from pcd.config.config import config
import utils
import h5py

class Grid_Visualiser():
    def __init__(self, rows, cols, numsteps, row_headings=None, xtitles=None, ytitles=None, norm=None, pred=False):

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
        self.pred = pred

        if not self.pred:
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

        if not self.pred:
            epoch = step * config['train']['cache_freq'] / (int(self.num_train / config['train']['batch_size']) + int(
                self.num_test / config['train']['batch_size']))

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

        self.fig.canvas.draw()
        # input("Press n to create a new figure or anything else to continue using this one")

    # plt.show(block=True)


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

def get_metric_distributions(cur_seg, pred_seg_res, sum=True):
    true_neg = np.logical_and(cur_seg == 0, np.round(pred_seg_res) == 0)  # round just in case the pred_val is in mean mode
    true_pos = np.logical_and(cur_seg == 1, np.round(pred_seg_res) == 1)
    false_neg = np.logical_and(cur_seg == 1, np.round(pred_seg_res) == 0)
    false_pos = np.logical_and(cur_seg == 0, np.round(pred_seg_res) == 1)

    metrics = [true_pos, false_neg, false_pos, true_neg]

    if sum:
        metrics = [int(np.sum(metric)) for metric in metrics]

    return metrics

def initialize_axes(metric_types):
    fig, axes = utils.init_grid(rows=2, cols=int(np.ceil(len(metric_types)/2)), figsize=(16, 9))
    axes = axes.flatten()
    for ax, metric_type in zip(axes, metric_types):
        ax.set_title(metric_type)
    return axes

def initialize_metrics(metric_types):
    values = [np.empty(0)]*len(metric_types)

    metrics = {}
    metrics['train'] = dict(zip(metric_types, values))
    metrics['test'] = dict(zip(metric_types, values))
    print(metrics['train'])
    metrics['train']['lines'], metrics['test']['lines'] = [], []
    metrics['train']['color'], metrics['test']['color'] = 'C1', 'C0'

    return metrics

def update_metrics(cur_seg, pred_seg_res, train, metrics, loss=-1):
    metrics_vol = get_metric_distributions(cur_seg, pred_seg_res, sum=False)

    # np.float64 so ZeroDivideErrors -> np.nan
    true_pos, false_neg, false_pos, true_neg = np.float64(np.sum(metrics_vol[0])), np.float64(np.sum(metrics_vol[1])), \
                                               np.float64(np.sum(metrics_vol[2])), np.float64(np.sum(metrics_vol[3])),

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

def plot_snr_trends(start=0, end=-1):
    alldata = load_meta('snr_data')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    snrdata = np.zeros((allsteps, len(alldata[0])))
    for step in range(start, end+1):
        dprint(step)

        snrdata[step] = alldata[step]

    metric_types = ['pix_snr', 'pix_signal', 'pix_back_mean', 'pix_back_std']
    axes = initialize_axes(metric_types)
    for i, (ax, metric) in enumerate(zip(axes, metric_types)):
        ax.plot(snrdata[:,i])
        ax.set_title(metric)
    plt.show()

def plot_fluxes(start=0, end=-1):
    alldata = load_meta('images')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    images = []

    for step in range(start, end+1):
        # plt.plot(alldata[step][1], label=f'{step}')
        images.append(alldata[step])

    plt.legend()
    fig, axes = utils.init_grid(rows=5, cols=5, figsize=(16,8))
    axes = axes.flatten()
    for im, image in enumerate(images[::4]):
        axes[im].imshow(image, origin='lower')
    plt.tight_layout()
    plt.show()


def onetime_metric_streams(start=0, end=10):
    """
    Shows metrics as a function of training steps

    :return:
    """

    plot_metric_types = ['True Positive', 'True Negative']
    metric_types = ['True Positive', 'False Negative', 'True Negative', 'False Positive', 'Recall', 'Precision',
                    'Accuracy', 'Loss']

    num_train, num_test = num_input()
    alldata = load_meta('pt_outputs')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)

    axes = initialize_axes(plot_metric_types)
    metrics = initialize_metrics(metric_types)

    for step in range(start, end+1):
        dprint(step)

        cur_seg, pred_seg_res, _, loss, train = alldata[step]
        dprint(train)

        metrics = update_metrics(cur_seg, pred_seg_res, train, metrics, loss)

    for kind in ['train', 'test']:

        # get the number of x values
        num = len(metrics[kind][plot_metric_types[0]])
        if kind == 'train':
            epochs = np.linspace(0, num * config['train']['cache_freq']/num_train, num)
        if kind == 'test':
            epochs = np.linspace(0, num * config['train']['cache_freq']/num_test, num)

        for ax, metric_key in zip(axes, plot_metric_types):
            metrics[kind]['lines'].append(ax.plot(epochs, metrics[kind][metric_key], c=metrics[kind]['color'],
                                                  label=kind))

    axes[0].legend()

    plt.show(block=True)

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

def metric_tesseracts(start=-50, end=-1, jump=1, type='both'):
    """ Shows the net predictions on the cloud as a series of 2d histograms in 4x4 grid of form
     ___________________________
    |_______|____coord_pairs____|
    |_layer_|_xy_|_py_|_ty_|_pt_|
    |___TP__|____|____|____|____|
    |___FP__|____|____|____|____|
    |___TN__|____|____|____|____|
    |___FN__|____|____|____|____|

    """

    assert end != 0
    # assert jump >= config['train']['cache_freq']
    alldata = load_meta('pt_outputs')
    allsteps = len(alldata)
    start, end = get_range_inds(start, end, allsteps)
    dprint(start, end)

    visualiser = Grid_Visualiser(4, 4, row_headings= ['True Planet','Missed Planet','True Star','Missed Star'],
                                 xtitles=['X','Phase','Time','Phase'], ytitles=['Y']*4, numsteps=allsteps,
                                 norm=LogNorm(), pred=type=='eval')

    if config['data']['quantize']:
        _,_, cur_data, _, _ = alldata[0]
        bins = [np.linspace(np.min(cur_data[:,0]),np.max(cur_data[:,0]), 100)] * 4
    else:
        bins = [np.linspace(-1, 1, 100) * 1e6] * 4

    dim_pairs = [[3, 2], [3, 1], [3, 0], [1, 2]]

    for step in range(start, end+1, jump):
        cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[step]

        metrics = get_metric_distributions(cur_seg, pred_seg_res, sum=False)
        true_pos, false_neg, false_pos, true_neg = np.sum(metrics, axis=1)

        print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))

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
        visualiser.update(step, trainbool, images, norm=None, extent=[-1,1,-1,1])
    plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--epoch', default=-1, dest='epoch', help='View the performance of which epoch')
    args = parser.parse_args()
    plot_fluxes()
    plot_snr_trends()
    # onetime_metric_streams(end = -1)
    # metric_tesseracts(start = 0, end = -1, jump=1)

