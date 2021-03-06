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
from vip_hci.metrics import contrcurve, aperture_flux, noise_per_annulus, snrmap, snr
from vip_hci.preproc import cube_derotate
from medis.plot_tools import grid

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config
from pcd.input import load_h5
from pcd.utils import load_meta, get_bin_measures, confusion_matrix
from pcd.analysis import calc_snr, get_photons, get_tess, reduce_image, find_loc, pix_snr_loc

home = os.path.expanduser("~")
# sp.numframes = 100
# sp.sample_time = 0.05

def get_reduced_images(ind=1, use_spec=True, plot=False, use_pca=True, verbose=False):
    """ Looks for reduced images file in the home folder and returns if it exists. If you're on the local machine
    and the file has not been transferred it will throw a FileNotFoundError """

    all_photons, star_photons, planet_photons = get_photons(amount=-ind)
    all_tess, star_tess, planet_tess = (get_tess(photons) for photons in [all_photons, star_photons, planet_photons])

    all_tess = np.transpose(all_tess, axes=(1,0,2,3))
    star_tess = np.transpose(star_tess, axes=(1,0,2,3))
    planet_tess = np.transpose(planet_tess, axes=(1,0,2,3))

    all_raw = np.sum(all_tess, axis=(0,1))
    star_raw = np.sum(star_tess, axis=(0,1))
    planet_raw = np.sum(planet_tess, axis=(0,1))

    sp.numframes = 80
    sp.sample_time = 15

    angle_list = -np.linspace(0, sp.numframes * sp.sample_time * config['data']['rot_rate']/60, all_tess.shape[1])
    print(angle_list, 'angle')

    if use_spec:
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], all_tess.shape[0])
        scale_list = wsamples / (ap.wvl_range[1] - ap.wvl_range[0])

        all_pca = pca.pca(all_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=(None,1), collapse='sum', verbose=verbose)

        star_pca = pca.pca(star_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=(None,1), collapse='sum', verbose=verbose)

        planet_pca = pca.pca(planet_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=(None,1), collapse='sum', verbose=verbose)
        # reduced_images = np.array([[all_raw, star_raw, planet_raw], [all_pca, star_pca, planet_pca]])


        all_med = medsub_source.median_sub(all_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        star_med = medsub_source.median_sub(star_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        planet_med = medsub_source.median_sub(planet_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')

    else:
        pass
        # all_med = medsub_source.median_sub(np.sum(all_tess, axis=0), angle_list=angle_list, collapse='sum')
        # star_med = medsub_source.median_sub(np.sum(star_tess, axis=0), angle_list=angle_list, collapse='sum')
        # planet_med = medsub_source.median_sub(np.sum(planet_tess, axis=0), angle_list=angle_list,
        #                                       collapse='sum')

    all_derot = np.sum(cube_derotate(np.sum(all_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)
    star_derot = np.sum(cube_derotate(np.sum(star_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)
    planet_derot = np.sum(cube_derotate(np.sum(planet_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)

    # all_snr = snrmap(all_pca, fwhm=5, plot=True)
    # star_snr = snrmap(star_pca, fwhm=5, plot=True)
    # planet_snr = snrmap(planet_pca, fwhm=5, plot=True)

    reduced_images = np.array([[all_raw, star_raw, planet_raw],
                               [all_derot, star_derot, planet_derot],
                               [all_pca, star_pca, planet_pca],
                               [all_med, star_med, planet_med]])#, )  # [all_snr, star_snr, planet_snr]

    planet_loc = (92,60)
    fwhm =  config['data']['fwhm']
    nproc = 8

    # reduced_images = np.array([[all_med, snrmap(all_med, fwhm, known_sources=planet_loc, nproc=nproc)],
    #                            [all_pca, snrmap(all_pca, fwhm, known_sources=planet_loc,  nproc=nproc)],
    #                            [planet_derot, snrmap(planet_derot, fwhm, known_sources=planet_loc, nproc=nproc)],
    #                            [planet_pca, snrmap(planet_pca, fwhm, known_sources=planet_loc,  nproc=nproc)]])
    # plt.imshow(planet_derot)
    # plt.imshow(snrmap(planet_derot, fwhm, nproc=nproc))
    # plt.show()
    # reduced_images = np.array([[all_raw, star_raw, planet_raw]])

    # grid(reduced_images, logZ=True, vlim=(1,50))  #, vlim=(1,70)
    if plot:
        grid(reduced_images, vlim=(-10,106))

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
    cur_seg, pred_seg_res, cur_data, _, trainbool, _ = alldata[-5]
    del alldata

    cur_data[:,0] = (cur_data[:,0] + 200) * 30/400
    cur_data[:,2] = (cur_data[:,2] + 200) * mp.array_size[0]/400
    cur_data[:,3] = (cur_data[:,3] + 200) * mp.array_size[0]/400

    true_pos, false_neg, false_pos, true_neg = get_bin_measures(cur_seg, pred_seg_res, sum=True)

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

# def ROC_curve():
#     amount = -1
#     print('amount = ', amount)
#     alldata = load_meta(kind='pt_outputs', amount=amount)
#     cur_seg, pred_seg_res, cur_data, _, trainbool, _ = alldata[-amount]
#     del alldata
#
#     from sklearn.metrics import roc_curve, auc
#     fpr, tpr, _ = roc_curve(cur_seg,pred_seg_res[:,1])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc="lower right")
#     plt.show()

def view_reduced():
    trios = []
    for i in range(6,21,1):
        reduced_images = get_reduced_images(ind=-i, plot=False)
        PCA = reduced_images[1,0]
        PCD = reduced_images[0,2]
        PCDPCA = reduced_images[1,2]
        trio = np.array([PCA, PCD, PCDPCA])
        trio[trio<=0] = 0.1
        trios.append(trio)

        if i % 5 == 0:
            print(i)
            grid(trios, logZ=True, vlim=(1,60), show=False)
            trios = []

    plt.show(block=True)

def contrast_curve():
    thrus = np.zeros((4,5,3))  # 4 conts, 5 rad, 3 types
    r = range(73)
    stds = np.zeros((1,len(r),3))
    fwhm =  config['data']['fwhm']
    planet_locs = np.array([[39,36],[46,44],[53,51],[60,59],[66,65]])
    loc_rads = np.sqrt(np.sum((mp.array_size[0]//2-planet_locs)**2, axis=1))

    imlistsfile = 'imlists.npy'
    if os.path.exists(imlistsfile):
        with open(imlistsfile, 'rb') as f:
            imlists = np.load(f)
    else:
        imlists = []
        alldata = load_meta(kind='pt_outputs')
        for ix, rev_ind in enumerate(range(1, 21, 1)): #1,21

            cur_seg, pred_seg_res, cur_data, _, trainbool, _ = alldata[-rev_ind]
            metrics = get_bin_measures(cur_seg, pred_seg_res, sum=False)

            true_star_photons = np.concatenate((cur_data[metrics[1]], cur_data[metrics[3]]), axis=0)

            bins = [np.linspace(true_star_photons[:, 0].min(), true_star_photons[:, 0].max(), sp.numframes + 1),
                    np.linspace(true_star_photons[:, 1].min(), true_star_photons[:, 1].max(), ap.n_wvl_final + 1),
                    np.linspace(-200, 200, 151),
                    np.linspace(-200, 200, 151)]

            true_star_tess, edges = np.histogramdd(true_star_photons, bins=bins)

            true_star_tess = np.transpose(true_star_tess, axes=(1, 0, 2, 3))
            angle_list = -np.linspace(0, sp.numframes * sp.sample_time * config['data']['rot_rate']/60, true_star_tess.shape[1])
            star_derot = np.sum(
                cube_derotate(np.sum(true_star_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)

            reduced_images = get_reduced_images(ind=-rev_ind, plot=False)

            PCA = reduced_images[1, 0]
            PCD = reduced_images[0, 2]
            PCDPCA = reduced_images[1, 2]

            imlist = np.array([star_derot, PCA, PCD, PCDPCA])
            # imlist[imlist <= 0] = 0.01
            imlists.append(imlist)

        with open(imlistsfile, 'wb') as f:
            np.save(f, np.array(imlists))

    trios = []
    for ix in range(20):
        imlist = imlists[ix]
        star_derot, PCA, PCD, PCDPCA = imlist
        cont_ind = ix //5
        r_ind = ix % 5

        trio = np.array([PCA, PCD, PCDPCA])
        trio[trio<=0] = 0.1
        trios.append(trio)

        if ix % 5 == 4:
            print(ix)
            grid(trios, logZ=True, vlim=(1,60), show=False)
            trios = []

        plot = False
        # if cont_ind >= 0:
        #     plot = True

        # true_flux = aperture_flux(planet_derot, [planet_locs[r_ind,0]],[planet_locs[r_ind,1]], fwhm, plot=plot)[0]
        # measured = np.array([aperture_flux(PCA, [planet_locs[r_ind,0]],[planet_locs[r_ind,1]], fwhm, plot=plot)[0],
        #                      aperture_flux(PCD, [planet_locs[r_ind,0]], [planet_locs[r_ind,1]], fwhm, plot=plot)[0],
        #                      aperture_flux(PCDPCA, [planet_locs[r_ind,0]], [planet_locs[r_ind,1]], fwhm, plot=plot)[0]
        #                      ])


        # print(ix, cont_ind, r_ind, measured/true_flux, true_flux)
        # thrus[cont_ind, r_ind] = measured/true_flux

        if cont_ind == 0: #4-1:
            std = np.array([noise_per_annulus(PCA, 1, fwhm)[0],
                            noise_per_annulus(PCD, 1, fwhm)[0],
                            noise_per_annulus(PCDPCA, 1, fwhm)[0]
                            ])
            stds[0] = std.T

    plt.show()

    # mean_thrus = np.mean(thrus[2:], axis=0)
    full_thru = noise_per_annulus(star_derot, 1, fwhm, mean_per_ann=True)[0]
    # full_thrus = []
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    labels = ['PCA', 'PCD (sum)', 'PCD + PCA']
    for i in range(3):
        # ax1.plot(loc_rads[::-1], mean_thrus[:,i][::-1], 'o')
        # f = InterpolatedUnivariateSpline(loc_rads[::-1], mean_thrus[:,i][::-1], k=1, ext=3)  # switch order since furthest out is first
        # full_thru = f(r)
        # full_thru[full_thru<=0] = 0.01

        sensitivity = stds[0,:,i] /max(full_thru)
        ax1.plot(full_thru)
        ax2.plot(stds[0,:,i])
        ax3.plot(sensitivity)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")

        # plt.plot(sensitivity/1e5, label=labels[i])
    # plt.yscale("log")
    # plt.legend()
    plt.show(block=True)

        # grid(imlist, logZ=True, vlim=(1,60), show=False)
    # plt.show(block=True)

def analyze_saved(plot_images=False):
    """ image loading analysis with separation into train/test """
    alldata = load_meta('pt_outputs')
    allsteps = len(alldata)
    snrs = np.zeros(allsteps)
    trains = np.zeros(allsteps)
    tps = np.zeros(allsteps)
    tns = np.zeros(allsteps)

    for step in range(allsteps):
        true_label, pred_label, input_data, loss, train, astro_dict = alldata[step]
        tp_list, fn_list, fp_list, tn_list = get_bin_measures(true_label, pred_label, sum=False)
        planet_photons = np.concatenate((input_data[tp_list], input_data[fp_list]), axis=0)
        snr = calc_snr(planet_photons, astro_dict, plot=plot_images)
        snrs[step] = snr
        trains[step] = train

        tp_amount, fn_amount, fp_amount, tn_amount = np.sum([tp_list, fn_list, fp_list, tn_list], axis=1)
        tps[step] = tp_amount / (tp_amount + fn_amount)
        tns[step] = tn_amount / (tn_amount + fp_amount)

    test_snrs, train_snrs = snrs[trains == 0], snrs[trains == 1]
    test_tps, train_tps  = tps[trains == 0], tps[trains == 1]
    test_tns, train_tns = tns[trains == 0], tns[trains == 1]

    metric_list = [test_tps, train_tps, test_tns, train_tns, test_snrs, train_snrs]
    metric_tups = [(metric.mean(), metric.std() / np.sqrt(len(metric))) for metric in metric_list]

    return metric_tups

def pt_step(input_data, input_label, pred_val, loss, astro_dict, train=True, verbose=True):
    if not config['train']['roc_probabilities']:
        pred_val = np.argmax(pred_val, axis=-1)

    with open(config['train']['pt_outputs'], 'ab') as handle:
        field_tup = (input_label, pred_val, input_data, loss, train, astro_dict)
        pickle.dump(field_tup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        if config['train']['roc_probabilities']:
            pred_val = np.argmax(pred_val, axis=-1)

        metrics = get_bin_measures(input_label, pred_val, sum=False)
        true_pos, false_neg, false_pos, true_neg = np.sum(metrics, axis=1)
        conf = confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg)
        print(conf)
        print('throughput: ', true_pos  / (true_pos + false_neg))

        # planet_photons = np.concatenate((input_data[metrics[0]], input_data[metrics[2]]), axis=0)
        # calc_snr(planet_photons, astro_dict)

if __name__ == '__main__':
    get_reduced_images(ind=435, plot=True)
    # plot_3D_pointclouds()
    # ROC_curve()
    # view_reduced()
    # contrast_curve()
