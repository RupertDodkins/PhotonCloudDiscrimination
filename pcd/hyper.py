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

def points_performance():
    num_points = config['num_point']*np.array([1e-4,0.1,0.25,0.5,0.75,1])
    savepth = 'points_{}.pth'
    pt_out = 'pt_points_{}.pkl'
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
    epochs = np.arange(1,5)
    savepth = 'epochs/step_{}.pth'
    pt_out = 'epochs/pt_step_{}.pkl'
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

def plot_hype(x, stats, xtitle):
    stats[np.isnan(stats)] = 0

    metric_types = ['True Positive', 'True Negative', 'SNR']
    fig, axes = init_grid(rows=len(metric_types), cols=1, figsize=(9, 16))
    axes = axes.flatten()

    for im, (ax, metric) in enumerate(zip(axes, metric_types)):
        istat = im * 2
        ax.errorbar(x, stats[:,istat,0], yerr=stats[:,istat,1], label='test') #,
        ax.errorbar(x, stats[:,istat+1,0], yerr=stats[:,istat+1,1], label='train') #,
        ax.set_ylabel(metric)
        ax.set_xlabel(xtitle)

    ax.legend()
    plt.tight_layout()
    plt.show()


def blob_ROC_curves():
    predict()
    reduced_images = get_reduced_images(ind=-1, plot=False)
    PCDPCA = reduced_images[1, 2]
    star_derot = reduced_images[0, 0]

if __name__ == '__main__':
    if not os.path.exists(config['working_dir']):
        make_input(config)

    # points_performance()
    # contrast_performance()
    epoch_performance()
    # input_performance()
