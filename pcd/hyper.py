import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import copy

from pcd.input import make_input
from pcd.train import train, load_dataset
from pcd.predict import predict
from pcd.article_plots import get_reduced_images, analyze_saved
from pcd.config.config import config
from pcd.utils import init_grid

def weights_performance():
    weight_ratios = [1e-6, 1e-5, 1e-4, 0.001, 0.01, 1.]
    savepth = 'weightsredo/weights_{}.pth'
    pt_out = 'weightsredo/pt_weights_{}.pkl'
    stats = np.zeros((len(weight_ratios),6,2))

    for r, ratio in enumerate(weight_ratios):
        config['savepath'] = config['working_dir']+savepth.format(ratio)
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(ratio)

        if not os.path.exists(config['savepath']):
            config['train']['max_epoch'] = 4
            config['weight_ratio'] = ratio
            train(verbose=False)
        stats[r] = analyze_saved(plot_images=False)

    plot_hype(weight_ratios, stats, 'Num points', logx=True)

def points_performance():
    num_points = config['num_point']*np.array([1e-4,0.1,0.25,0.5,0.75,1])
    savepth = 'points4epoch/points_{}.pth'
    pt_out = 'points4epoch/pt_points_{}.pkl'
    stats = np.zeros((len(num_points),6,2))

    for p, point in enumerate(num_points):
        config['savepath'] = config['working_dir']+savepth.format(point)
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(point)

        if not os.path.exists(config['savepath']):
            config['train']['max_epoch'] = 4
            config['data']['degrade_factor'] = config['num_point']/point
            train(verbose=False)
        stats[p] = analyze_saved(plot_images=False)

    plot_hype(num_points, stats, 'Num points')

def contrast_performance():
    contrasts = [-2,-3,-4]
    savepth = 'cont_{}.pth'
    pt_out = 'pt_cont_{}.pkl'
    stats = np.zeros((len(contrasts),6,2))

    file_contrasts = []
    all_train = np.array(copy.copy(config['trainfiles']))
    num_test = int(copy.copy(config['data']['num_indata'] * config['data']['test_frac']))

    for i in range(int(config['data']['num_indata'] * (1 - config['data']['test_frac']))):
        _, _, astro_dict = load_dataset(config['trainfiles'][i])
        file_contrasts.append(astro_dict['contrast'])

    print(file_contrasts)
    file_bools = [np.logical_and(cont-0.5<=np.log10(file_contrasts), np.log10(file_contrasts)<cont+0.5) for cont in contrasts]

    for c, cont in enumerate(contrasts):
        config['savepath'] = config['working_dir']+savepth.format(cont)
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(cont)

        if not os.path.exists(config['savepath']):
            config['trainfiles'] = all_train[file_bools[c]]
            config['train']['max_epoch'] = 2
            config['data']['num_indata'] = len(config['trainfiles']) + num_test
            config['data']['test_frac'] = num_test/config['data']['num_indata']
            train(verbose=False)

        stats[c] = analyze_saved()

    plot_hype(contrasts, stats, 'Input contrast')

def epoch_performance():
    epochs = np.arange(1,4,1)
    savepth = 'epochsnotrack/epoch_{}.pth'
    pt_out = 'epochsnotrack/pt_epoch_{}.pkl'
    stats = np.zeros((len(epochs),6,2))

    for s in range(len(epochs)):
        config['savepath'] = config['working_dir']+savepth.format(epochs[s])
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(epochs[s])

        if not os.path.exists(config['savepath']):
            if s>0:
                prevpth = config['working_dir'] + savepth.format(epochs[s-1])
                print(f"starting training for {config['savepath']} with {prevpth}")
                shutil.copy(prevpth, config['savepath'])
                config['train']['max_epoch'] = epochs[s] - epochs[s-1]
            else:
                config['train']['max_epoch'] = epochs[s]
            train(verbose=True)

        stats[s] = analyze_saved()

    plot_hype(epochs, stats, 'Num epochs')

def input_performance():
    num_train = np.arange(1,int(config['data']['num_indata']*(1-config['data']['test_frac'])),2)
    num_train = np.arange(1,36,4)
    savepth = 'num_{}.pth'  #second_pt_num
    pt_out = 'pt_num_{}.pkl'  #second_pt_num/
    stats = np.zeros((len(num_train),6,2))
    all_train = copy.copy(config['trainfiles'])
    num_test = int(copy.copy(config['data']['num_indata']*config['data']['test_frac']))

    for n in range(len(num_train)):
        config['savepath'] = config['working_dir'] + savepth.format(num_train[n])
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(num_train[n])

        if not os.path.exists(config['savepath']):
            # config['train']['max_epoch'] = 2 # int(np.round(num_in.max()/num_in[n]))
            config['trainfiles'] = all_train[:num_train[n]]
            config['data']['num_indata'] = num_train[n] + num_test
            config['data']['test_frac'] = num_test/config['data']['num_indata']
            train(verbose=True)

        stats[n] = analyze_saved()

    plot_hype(num_train, stats, 'Num input')

def plot_hype(x, stats, xtitle, showylabel=True, logx=False):
    stats[np.isnan(stats)] = 0
    metric_types = ['True Positive', 'True Negative', 'SNR']
    xsize = 3.3 if showylabel else 3
    left = 0.18 if showylabel else 0.12

    fig, axes = plt.subplots(nrows=len(metric_types), ncols=1, sharex=True, figsize=(xsize, 16))
    axes = axes.flatten()

    for im, (ax, metric) in enumerate(zip(axes, metric_types)):
        istat = im * 2
        ax.errorbar(x, stats[:,istat,0], yerr=stats[:,istat,1], label='test') #,
        ax.errorbar(x, stats[:,istat+1,0], yerr=stats[:,istat+1,1], label='train') #,
        ax.tick_params(axis="x", direction="inout")
        ax.set_xscale('log')
        if showylabel:
            ax.set_ylabel(metric)

    ax.set_xlabel(xtitle)
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0, bottom=.08, left=left)
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(config['working_dir']):
        make_input(config)

    # points_performance()
    # contrast_performance()
    # epoch_performance()
    # input_performance()
    weights_performance()
