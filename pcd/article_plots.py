"""This file supports two python interpretors. One on the remote GPU machine (medis-tf) and one om the local machine
(pipeline-new)"""


import os
import pickle
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from vip_hci import pca
from vip_hci.medsub import medsub_source
from vip_hci.metrics import contrcurve, aperture_flux, noise_per_annulus
from vip_hci.preproc import cube_derotate
from medis.params import ap, sp, mp, tp
from medis.plot_tools import grid

from visualization import load_meta, get_metric_distributions, confusion_matrix
from pcd.config.config import config
import utils

home = os.path.expanduser("~")
# sp.numframes = 100
# sp.sample_time = 0.05

def get_reduced_images(ind=1, use_spec=True, plot=False, use_pca=True, verbose=False):
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

    angle_list = -np.linspace(0, sp.numframes * sp.sample_time * config['data']['rot_rate']/60, all_tess.shape[1])


    if use_spec:
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], all_tess.shape[0])
        scale_list = wsamples / (ap.wvl_range[1] - ap.wvl_range[0])

        all_pca = pca.pca(all_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=None, ncomp2=1, collapse='sum', verbose=verbose)

        star_pca = pca.pca(star_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=None, ncomp2=1, collapse='sum', verbose=verbose)

        planet_pca = pca.pca(planet_tess, angle_list=angle_list, scale_list=scale_list, mask_center_px=None,
                        adimsdi='double', ncomp=None, ncomp2=1, collapse='sum', verbose=verbose)
        # reduced_images = np.array([[all_raw, star_raw, planet_raw], [all_pca, star_pca, planet_pca]])


        # all_med = medsub_source.median_sub(all_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        # star_med = medsub_source.median_sub(star_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')
        # planet_med = medsub_source.median_sub(planet_tess, angle_list=angle_list, scale_list=scale_list, collapse='sum')

    else:
        pass
        # all_med = medsub_source.median_sub(np.sum(all_tess, axis=0), angle_list=angle_list, collapse='sum')
        # star_med = medsub_source.median_sub(np.sum(star_tess, axis=0), angle_list=angle_list, collapse='sum')
        # planet_med = medsub_source.median_sub(np.sum(planet_tess, axis=0), angle_list=angle_list,
        #                                       collapse='sum')

    all_derot = np.sum(cube_derotate(np.sum(all_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)
    star_derot = np.sum(cube_derotate(np.sum(star_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)
    planet_derot = np.sum(cube_derotate(np.sum(planet_tess, axis=0), angle_list, imlib='opencv', interpolation='lanczos4'), axis=0)

    reduced_images = np.array([[all_derot, star_derot, planet_derot], [all_pca, star_pca, planet_pca]])#, [all_med, star_med, planet_med]])
    # reduced_images = np.array([[all_raw, star_raw, planet_raw]])

    # grid(reduced_images, logZ=True, vlim=(1,50))  #, vlim=(1,70)
    if plot:
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

    true_pos, false_neg, false_pos, true_neg = get_metric_distributions(cur_seg, pred_seg_res, sum=True)

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

    metrics = get_metric_distributions(cur_seg, pred_seg_res, sum=False)
    true_pos, false_neg, false_pos, true_neg = np.sum(metrics, axis=1)
    print(trainbool)
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

    # bins = [np.linspace(-1, 1, sp.numframes + 1), np.linspace(-1, 1, ap.n_wvl_final + 1),
    #         np.linspace(-1, 1, mp.array_size[0]), np.linspace(-1, 1, mp.array_size[1])]
    # bins = [150] * 4
    bins = [np.linspace(all_photons[:,0].min(), all_photons[:,0].max(), sp.numframes + 1),
            np.linspace(all_photons[:,1].min(), all_photons[:,1].max(), ap.n_wvl_final + 1),
            np.linspace(-200, 200, 151),
            np.linspace(-200, 200, 151)]

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
    fwhm = 5 #8.5
    planet_locs = np.array([[39,36],[46,44],[53,51],[60,59],[66,65]])
    loc_rads = np.sqrt(np.sum((75-planet_locs)**2, axis=1))

    imlistsfile = 'imlists.npy'
    if os.path.exists(imlistsfile):
        with open(imlistsfile, 'rb') as f:
            imlists = np.load(f)
    else:
        imlists = []
        alldata = load_meta(kind='pt_outputs')
        for ix, rev_ind in enumerate(range(1, 21, 1)): #1,21

            cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[-rev_ind]
            metrics = get_metric_distributions(cur_seg, pred_seg_res, sum=False)

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


def rad_cont():
    """  """
    fwhm = 5
    hwhm = fwhm/2
    image_width = 150
    alldata = load_meta(kind='pt_outputs')
    for i in range(3):
        cur_seg, pred_seg_res, cur_data, _, trainbool = alldata[-1]
        # del alldata

        metrics = get_metric_distributions(cur_seg, pred_seg_res, sum=False)
        true_pos, false_neg, false_pos, true_neg = np.sum(metrics, axis=1)

        tot_neg = true_neg + false_pos
        tot_pos = true_pos + false_neg
        print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, tot_neg, tot_pos))
        exo_pred = (true_pos + false_neg)/tot_pos
        if exo_pred == 0 or exo_pred == 1:
            cont = 1
            return cont

        throughput = true_pos/tot_pos

        planet_photons = np.concatenate((cur_data[metrics[0]], cur_data[metrics[2]]), axis=0)

        planet_map, _, _ = np.histogram2d(planet_photons[:, 3], planet_photons[:, 2],
                                          bins=[np.linspace(-200, 200, image_width)] * 2)

        raw_planet_location = np.mean(planet_photons[:,2:], axis=0)
        planet_location = (raw_planet_location+200) * image_width/400
        planet_radius = np.sqrt(np.sum(np.abs(image_width/2-planet_location)**2))

        star_photons = np.concatenate((cur_data[metrics[1]], cur_data[metrics[3]]), axis=0)
        star_map, _, _ = np.histogram2d(star_photons[:,3], star_photons[:,2], bins=[np.linspace(-200,200,image_width)]*2)
        stds, rads = noise_per_annulus(star_map, fwhm, fwhm)
        ann_ind = np.where((rads > planet_radius - hwhm) & (rads < planet_radius+ hwhm))[0][0]
        std = stds[ann_ind]

        print(raw_planet_location, planet_location, planet_radius, ann_ind)

        cont = std*5/(throughput*tot_neg)
        return cont

if __name__ == '__main__':
    get_reduced_images(ind=-8, plot=True)
    # plot_3D_pointclouds()
    # ROC_curve()
    # view_reduced()
    contrast_curve()

