"""This file supports two python interpretors. One on the remote GPU machine (medis-tf) and one om the local machine
(pipeline-new)"""


import os
import pickle
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
import numpy as np
from vip_hci import pca
from vip_hci.medsub import medsub_source
from vip_hci.metrics import contrcurve
from medis.params import ap, sp, mp, tp
from medis.plot_tools import grid

from visualization import load_meta, get_metric_distributions, confusion_matrix, trans_p2c
from pcd.config.config import config

home = os.path.expanduser("~")
sp.numframes = 100
sp.sample_time = 0.05

def get_reduced_images(ind=1, use_spec=False):
    """ Looks for reduced images file in the home folder and returns if it exists. If you're on the local machine
    and the file has not been transferred it will throw a FileNotFoundError """
    # if home == '/Users/dodkins':
    #     filename = os.path.join(home, 'reduced_images.pkl')
    #     if os.path.exists(filename):
    #         with open(filename, 'rb') as handle:
    #             reduced_images = pickle.load(handle)
    #     else:
    #         print('No file found. Need to create transfer')
    #         raise FileNotFoundError
    # elif home == '/home/dodkins':
    #     filename = os.path.join(home, 'reduced_images.pkl')
    #     if os.path.exists(filename):
    #         with open(filename, 'rb') as handle:
    #             reduced_images = pickle.load(handle)
    #     else:

    all_tess, star_tess, planet_tess = get_tess(ind=ind)

    all_tess = np.transpose(all_tess, axes=(1,0,2,3))
    star_tess = np.transpose(star_tess, axes=(1,0,2,3))
    planet_tess = np.transpose(planet_tess, axes=(1,0,2,3))

    all_raw = np.sum(all_tess, axis=(0,1))
    star_raw = np.sum(star_tess, axis=(0,1))
    planet_raw = np.sum(planet_tess, axis=(0,1))

    angle_list = -np.linspace(0, 2 * sp.numframes * sp.sample_time * tp.rot_rate, all_tess.shape[1])

    if use_spec:
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], all_tess.shape[0])
        scale_list = wsamples / (ap.wvl_range[1] - ap.wvl_range[0])




        # all_pca = pca.pca(all_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
        #                 adimsdi='double', ncomp=None, ncomp2=1, collapse='sum')
        #
        # star_pca = pca.pca(star_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
        #                 adimsdi='double', ncomp=None, ncomp2=1, collapse='sum')
        #
        # planet_pca = pca.pca(planet_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
        #                 adimsdi='double', ncomp=None, ncomp2=1, collapse='sum')
        # reduced_images = np.array([[all_raw, star_raw, planet_raw], [all_pca, star_pca, planet_pca]])

        all_med = medsub_source.median_sub(all_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        star_med = medsub_source.median_sub(star_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        planet_med = medsub_source.median_sub(planet_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')

    else:
        all_med = medsub_source.median_sub(np.sum(all_tess, axis=0), angle_list=angle_list, collapse='sum')
        star_med = medsub_source.median_sub(np.sum(star_tess, axis=0), angle_list=angle_list, collapse='sum')
        planet_med = medsub_source.median_sub(np.sum(planet_tess, axis=0), angle_list=angle_list,
                                              collapse='sum')

    reduced_images = np.array([[all_raw, star_raw, planet_raw], [all_med, star_med, planet_med]])
    # reduced_images = np.array([[all_raw, star_raw, planet_raw]])

    # grid(reduced_images, logZ=True, vlim=(1,50))  #, vlim=(1,70)
    grid(reduced_images, vlim=(-9,40))  #, vlim=(1,70)

    return reduced_images

def plot_reduced_images(ind=-1):
    reduced_images = np.array(get_reduced_images(ind=ind))

    vmaxes = [[125, 5], [None, None]]
    texts = [['Raw', 'PCD'], ['PCA', 'PCD+PCA']]
    # fig, axs = plt.subplots(2, 2)
    # for col in range(2):
    #     for row in range(2):
    #         ax = axs[row, col]
    #         # pcm = ax.imshow([[all_raw, planet_raw], [all_pca, planet_pca]][col][row], vmax=vmaxes[col][row], origin='lower')
    #         pcm = ax.imshow(reduced_images[:,[0,2]][col][row], vmax=vmaxes[col][row], origin='lower')
    #         ax.text(15,15,texts[col][row],color='w', fontsize=15)
    #         fig.colorbar(pcm, ax=ax)
    # plt.tight_layout()
    # plt.savefig('PCD.pdf')
    # # plt.show(block=True)

    plt.figure()
    labels = np.array(texts).flatten()
    images = np.array(reduced_images[:,[0,2]]).reshape(4,149,149)
    for l, image in enumerate(images):
        noise_curve = contrcurve.noise_per_annulus(image, 3, 3)
        plt.plot(noise_curve[1], noise_curve[0], label=labels[l])

    # plt.yscale('log')
    plt.legend()
    plt.xlabel('Separation (pixels)')
    plt.ylabel('Contrast (currently annular std)')
    plt.show()

def plot_3D_pointclouds():
    alldata = load_meta()
    cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[-5]
    del alldata

    cur_data[:,0] = (cur_data[:,0] + 200) * 30/400
    cur_data[:,2] = (cur_data[:,2] + 200) * 150/400
    cur_data[:,3] = (cur_data[:,3] + 200) * 150/400

    metrics = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)

    true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics[0])), int(np.sum(metrics[1])), \
                                               int(np.sum(metrics[2])), int(np.sum(metrics[3])),

    print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))
    all_photons = np.concatenate(
        (cur_data[metrics[0]], cur_data[metrics[1]], cur_data[metrics[2]], cur_data[metrics[3]]),
        axis=0)
    star_photons = np.concatenate((cur_data[metrics[1]], cur_data[metrics[3]]), axis=0)
    planet_photons = np.concatenate((cur_data[metrics[0]], cur_data[metrics[2]]), axis=0)

    neg = cur_seg == 0
    pos = cur_seg == 1

    degrade_factor = 50

    true_star_photons = cur_data[neg][::degrade_factor]
    true_planet_photons = cur_data[pos][::degrade_factor]

    print(len(all_photons), len(true_star_photons), len(true_planet_photons), len(planet_photons))

    fig = plt.figure(figsize=(17,6))
    all_photons, star_photons, planet_photons = all_photons[::degrade_factor], star_photons[::degrade_factor], planet_photons[::degrade_factor]
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(30, -210)
    ax.set_xlabel('Time (m)')
    ax.set_ylabel('RA pixel')
    ax.set_zlabel('Dec pixel')
    # plt.axis('off')
    # ax.zaxis_inverted()

    ax.set_title('Raw Input')
    ax.scatter(all_photons[:,0], all_photons[:,2], all_photons[:,3], s=2, alpha=1, c='grey')

    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(30, -210)
    ax.set_xlabel('Time (m)')
    ax.set_ylabel('RA pixel')
    ax.set_zlabel('Dec pixel')
    # plt.axis('off')
    # ax.zaxis_inverted()
    ax.set_title('Predictions')
    ax.scatter(star_photons[:, 0], star_photons[:, 2], star_photons[:, 3], s=2, alpha=1)
    ax.scatter(planet_photons[:, 0], planet_photons[:, 2], planet_photons[:, 3], s=2, alpha=1, c='red')

    ax = fig.add_subplot(133, projection='3d')
    ax.view_init(30, -210)
    ax.set_xlabel('Time (m)')
    ax.set_ylabel('RA pixel')
    ax.set_zlabel('Dec pixel')
    # ax.zaxis_inverted()
    ax.set_title('Ground Truth')
    ax.scatter(true_star_photons[:, 0], true_star_photons[:, 2], true_star_photons[:, 3], s=2, alpha=1)
    ax.scatter(true_planet_photons[:, 0], true_planet_photons[:, 2], true_planet_photons[:, 3], s=2, alpha=1, c='red')
    # ax.scatter(star_photons[:, 0], star_photons[:, 2], star_photons[:, 3], s=1, alpha=0.1)
    # ax.scatter(planet_photons[:, 0], planet_photons[:, 2], planet_photons[:, 3], s=1, alpha=0.1, c='red')

    # for t in range(10):
    #     print(t)
    #     for i, c in enumerate(colors):
    #         # print(i, c, len(self.chunked_photons[t, (self.chunked_pids[t, :, 0] == i), 0][::downsamp]))
    #         ax.scatter(self.chunked_photons[t, (self.chunked_pids[t, :, 0] == i), 0][::downsamp],
    #                    self.chunked_photons[t, (self.chunked_pids[t, :, 0] == i), 1][::downsamp],
    #                    self.chunked_photons[t, (self.chunked_pids[t, :, 0] == i), 2][::downsamp], c=c,
    #                    marker='.')  # , marker=pids[0])
    plt.tight_layout()
    plt.show(block=True)

def get_photons(amount=1):
    print('amount = ', amount)
    alldata = load_meta(kind='pt_outputs', amount=amount)
    cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[-amount]
    del alldata

    metrics = get_metric_distributions(cur_seg, pred_seg_res, include_true_neg=True)
    true_pos, false_neg, false_pos, true_neg = int(np.sum(metrics[0])), int(np.sum(metrics[1])), \
                                               int(np.sum(metrics[2])), int(np.sum(metrics[3])),
    print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))

    # cur_data = cur_data[:, :, [0, 2, 1, 3]]

    # fig = plt.figure()
    all_photons = np.concatenate(
        (cur_data[metrics[0]], cur_data[metrics[1]], cur_data[metrics[2]], cur_data[metrics[3]]),
        axis=0)
    star_photons = np.concatenate((cur_data[metrics[1]], cur_data[metrics[3]]), axis=0)
    planet_photons = np.concatenate((cur_data[metrics[0]], cur_data[metrics[2]]), axis=0)

    return all_photons, star_photons, planet_photons

def get_tess(ind=-1):
    all_photons, star_photons, planet_photons = get_photons(amount=-ind)

    # all_photons, star_photons, planet_photons = all_photons[:,:-1], star_photons[:,:-1], planet_photons[:,:-1]
    # if config['data']['trans_polar']:
    #     for photons in [all_photons, star_photons, planet_photons]:
    #         photons = trans_p2c(photons)

    # bins = [np.linspace(-1, 1, sp.numframes + 1), np.linspace(-1, 1, ap.n_wvl_final + 1),
    #         np.linspace(-1, 1, mp.array_size[0]), np.linspace(-1, 1, mp.array_size[1])]
    # bins = [150] * 4
    bins = [np.linspace(all_photons[:,0].min(), all_photons[:,0].max(), sp.numframes + 1),
            np.linspace(all_photons[:,1].min(), all_photons[:,1].max(), ap.n_wvl_final + 1),
            np.linspace(-200, 200, 100),
            np.linspace(-200, 200, 100)]

    all_tess, edges = np.histogramdd(all_photons, bins=bins)

    star_tess, edges = np.histogramdd(star_photons, bins=bins)

    planet_tess, edges = np.histogramdd(planet_photons, bins=bins)

    return all_tess, star_tess, planet_tess

def ROC_curve():
    amount = -1
    print('amount = ', amount)
    alldata = load_meta(kind='pt_outputs', amount=amount)
    cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[-amount]
    del alldata

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(cur_seg,pred_seg_res[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def contrast_curve(): 
    pass


if __name__ == '__main__':
    # get_reduced_images(ind=-2)
    # plot_3D_pointclouds()
    ROC_curve()
