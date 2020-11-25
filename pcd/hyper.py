import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import copy

from pcd.input import make_input
from pcd.train import train
from pcd.predict import predict
from pcd.article_plots import get_reduced_images, snr_stats
from pcd.config.config import config

# def step_performance():
#     """
#     Must first create input files with input.py. config for making the input data is in PCD_data/201021
#
#     :return:
#     """
#
#     # config['testfiles'] = config['testfiles'][-1:]  # just select first example
#     config['testfiles'] = config['testfiles'][:1]
#
#     steps = [1,2,4,10,20,40]
#     fwhm = 5
#     fhqm = fwhm/2
#     lods = range(1,7)
#     savepth = 'step_{}.pth'
#     pt_out = 'pt_step_{}.pkl'
#
#     lod_conts = np.zeros((len(lods),len(steps)))
#     rad = np.arange(73)
#
#     thruput = np.array([0.01,1,2,2,3,3])/3.
#     for s in range(len(steps)):
#
#         # create model saves
#         config['savepath'] = config['working_dir']+savepth.format(steps[s])
#         if not os.path.exists(config['savepath']):
#             if s>0:
#                 prevpth = config['working_dir'] + savepth.format(steps[s-1])
#                 print(f"starting training for {config['savepath']} with {prevpth}")
#                 shutil.copy(prevpth, config['savepath'])
#                 config['train']['max_epoch'] = steps[s] - steps[s-1]
#             else:
#                 config['train']['max_epoch'] = steps[s]
#             train(verbose=True)
#         config['train']['pt_outputs'] = config['working_dir']+pt_out.format(steps[s])
#         if not os.path.exists(config['train']['pt_outputs']):
#             predict()
#
#         reduced_images = get_reduced_images(ind=-1, plot=False)
#         PCDPCA = reduced_images[1, 2]
#         star_derot = reduced_images[0, 0]
#
#         # get contrast
#         std = noise_per_annulus(PCDPCA, 1, fwhm)[0]
#         full_thru = noise_per_annulus(star_derot, 1, fwhm, mean_per_ann=True)[0]
#         sensitivity = std / (full_thru.max() * thruput[s])
#         sensitivity[sensitivity==0] = 1
#         print(thruput[s], 'thru')
#         plt.plot(sensitivity)
#
#         # measure contrast at 3 and 6L/D
#         for id, lod in enumerate(lods):
#             lod_ind = np.where((rad > lod*fwhm - fhqm) & (rad < lod*fwhm + fhqm))
#             lod_conts[id,s] = np.mean(sensitivity[lod_ind])
#     plt.yscale('log')
#     plt.figure()
#     for il, lod_cont in enumerate(lod_conts):
#         plt.plot(steps, lod_cont, label=f"{lods[il]}")
#     plt.xlabel('Training epochs')
#     plt.ylabel('Contrast')
#     plt.yscale('log')
#     plt.legend()
#     plt.show()

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

def plot_hype(x, snrs, xtitle=None):
    snrs[np.isnan(snrs)] = 0
    plt.errorbar(x, snrs[:,0], label='test') #, yerr=snrs[:,1]
    plt.errorbar(x, snrs[:,1], label='train')#, yerr=snrs[:,3]
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

    step_performance()
    # input_performance()
