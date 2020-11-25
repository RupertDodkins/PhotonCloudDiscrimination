import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import copy

from pcd.input import make_input
from pcd.train import train, load_dataset
from pcd.predict import predict
from pcd.article_plots import get_reduced_images, snr_stats
from pcd.config.config import config

def contrast_performance():
    contrasts = [-2,-3,-4]
    savepth = 'cont_{}.pth'
    pt_out = 'pt_cont_{}.pkl'
    snrs = np.zeros((len(contrasts),4))

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

        snrs[c] = snr_stats()

    plot_hype(contrasts, snrs, 'Input contrast')

def step_performance():
    steps = np.arange(1,12)
    savepth = 'step_{}.pth'
    pt_out = 'pt_step_{}.pkl'
    snrs = np.zeros((len(steps),4))

    for s in range(len(steps)):
        config['savepath'] = config['working_dir']+savepth.format(steps[s])
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(steps[s])

        if not os.path.exists(config['savepath']):
            if s>0:
                prevpth = config['working_dir'] + savepth.format(steps[s-1])
                print(f"starting training for {config['savepath']} with {prevpth}")
                shutil.copy(prevpth, config['savepath'])
                config['train']['max_epoch'] = steps[s] - steps[s-1]
            else:
                config['train']['max_epoch'] = steps[s]
            train(verbose=True)

        snrs[s] = snr_stats()

    plot_hype(steps, snrs, 'Num epochs')

def plot_hype(x, snrs, xtitle=None, plot_errs=True):
    snrs[np.isnan(snrs)] = 0
    if plot_errs:
        test_yerr, train_yerr = snrs[:, 1], snrs[:, 3]
    else:
        test_yerr, train_yerr = None, None
    plt.errorbar(x, snrs[:,0], yerr=test_yerr, label='test') #,
    plt.errorbar(x, snrs[:,1], yerr=train_yerr, label='train')#,
    plt.legend()
    if xtitle:
        plt.xlabel(xtitle)
    plt.ylabel('SNR')
    plt.legend()
    plt.show()

def input_performance():
    num_train = np.arange(1,int(config['data']['num_indata']*(1-config['data']['test_frac'])),2)
    # num_train = np.arange(1,28,2)
    savepth = 'num_{}.pth'  #second_pt_num
    pt_out = 'pt_num_{}.pkl'  #second_pt_num/
    snrs = np.zeros((len(num_train),4))
    all_train = copy.copy(config['trainfiles'])
    num_test = int(copy.copy(config['data']['num_indata']*config['data']['test_frac']))

    for n in range(len(num_train)):
        config['savepath'] = config['working_dir'] + savepth.format(num_train[n])
        config['train']['pt_outputs'] = config['working_dir'] + pt_out.format(num_train[n])

        if not os.path.exists(config['savepath']):
            config['train']['max_epoch'] = 2 # int(np.round(num_in.max()/num_in[n]))
            config['trainfiles'] = all_train[:num_train[n]]
            config['data']['num_indata'] = num_train[n] + num_test
            config['data']['test_frac'] = num_test/config['data']['num_indata']
            train(verbose=True)

        snrs[n] = snr_stats()

    plot_hype(num_train, snrs, 'Num input')

def blob_ROC_curves():
    predict()
    reduced_images = get_reduced_images(ind=-1, plot=False)
    PCDPCA = reduced_images[1, 2]
    star_derot = reduced_images[0, 0]

if __name__ == '__main__':
    if not os.path.exists(config['working_dir']):
        make_input(config)

    contrast_performance()
    # step_performance()
    # input_performance()
