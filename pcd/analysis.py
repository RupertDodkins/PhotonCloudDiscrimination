import matplotlib.pylab as plt
import numpy as np
from vip_hci.metrics import contrcurve, aperture_flux, noise_per_annulus, snrmap, snr
from vip_hci.preproc import cube_derotate

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config
from utils import load_meta, get_bin_measures, confusion_matrix, find_coords

def reduce_image(photons):
    tess = get_tess(photons)
    derot_image = derot_tess(tess)
    return derot_image

def find_loc(astro_dict, derot_image, verbose=False):
    planet_loc = astro_dict['loc'].astype('int') + mp.array_size // 2
    zoomim = derot_image[planet_loc[0] - 5:planet_loc[0] + 5, planet_loc[1] - 5:planet_loc[1] + 5]
    correction = np.array(np.unravel_index(np.argmax(zoomim), zoomim.shape)) - np.array([5, 5])
    planet_loc += correction

    if verbose:
        print('planet_loc: ', planet_loc)
    return planet_loc

def calc_snr(planet_photons, astro_dict, plot=False):
    derot_image = reduce_image(planet_photons)
    planet_loc = find_loc(astro_dict, derot_image)

    snr = pix_snr_loc(derot_image, planet_loc - mp.array_size // 2, config['data']['fwhm'], verbose=True)[0]
    # with open(config['train']['snr_data'], 'ab') as handle:
    #     pickle.dump(snr_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(config['train']['images'], 'ab') as handle:
    #     pickle.dump(derot_image, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if plot:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121)
        pcm = ax.imshow(derot_image, origin='lower')
        fig.colorbar(pcm, ax=ax)
        aper = plt.Circle(planet_loc[::-1], radius=config['data']['fwhm'] / 2., color='r', fill=False, alpha=0.8)
        ax.add_patch(aper)
        ax = fig.add_subplot(122)
        snrimage = snrmap(derot_image, fwhm=config['data']['fwhm'], nproc=8)
        pcm = ax.imshow(snrimage, origin='lower')
        fig.colorbar(pcm, ax=ax)
        plt.show()

    return snr

def pix_snr_loc(array, source_xy, fwhm, verbose=False):
    """
    homemade func for snr that differs from vips to account for jumps in number of apertures

    :param array:
    :param source_xy:
    :param fwhm:
    :param verbose:
    :param full_output:
    :return:
    """
    rad = np.sqrt(np.sum(source_xy**2))
    centered_coords = np.array([[],[]])

    for r in range(fwhm):
        centered_coords = np.concatenate((centered_coords, find_coords(rad-fwhm/2+r+0.5, 1)), axis=1)

    centered_coords = np.round(centered_coords).astype(int)
    offset = centered_coords - source_xy[:, np.newaxis]

    annuli_coords = centered_coords + mp.array_size[0]//2
    annuli_coords = np.delete(annuli_coords, np.where(np.sqrt(np.sum(offset ** 2, axis=0)) <= (fwhm/2)+4), axis=1)
    annuli_coords = np.delete(annuli_coords, np.where(np.logical_or(annuli_coords>=mp.array_size[0],annuli_coords<0)), axis=1)

    fluxes = array[annuli_coords[0],annuli_coords[1]]
    app_pix = np.pi*(fwhm/2)**2
    f_source = aperture_flux(array, [source_xy[0]+mp.array_size[0]//2], [source_xy[1]+mp.array_size[1]//2], fwhm)[0]/app_pix
    snr_value = (f_source-fluxes.mean())/fluxes.std()

    if verbose:
        msg1 = 'S/N for the given pixel = {:.3f}'
        msg2 = 'Integrated flux in FWHM test aperture = {:.3f}'
        msg3 = 'Mean of background apertures integrated fluxes = {:.3f}'
        msg4 = 'Std-dev of background apertures integrated fluxes = {:.3f}'
        print(msg1.format(snr_value))
        print(msg2.format(f_source))
        print(msg3.format(fluxes.mean()))
        print(msg4.format(fluxes.std()))

    return (snr_value, f_source, fluxes.mean(), fluxes.std())

def get_photons(amount=1):
    print('amount = ', amount)
    alldata = load_meta(kind='pt_outputs', amount=amount)
    cur_seg, pred_seg_res, cur_data, _, trainbool, _ = alldata[-amount]
    del alldata

    metrics = get_bin_measures(cur_seg, pred_seg_res, sum=False)
    true_pos, false_neg, false_pos, true_neg = np.sum(metrics, axis=1)
    print(trainbool)
    print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))

    all_photons = np.concatenate(
        (cur_data[metrics[0]], cur_data[metrics[1]], cur_data[metrics[2]], cur_data[metrics[3]]),
        axis=0)
    star_photons = np.concatenate((cur_data[metrics[1]], cur_data[metrics[3]]), axis=0)
    planet_photons = np.concatenate((cur_data[metrics[0]], cur_data[metrics[2]]), axis=0)

    return all_photons, star_photons, planet_photons

def get_tess(photonlist):
    # bins = [np.linspace(-1, 1, sp.numframes + 1), np.linspace(-1, 1, ap.n_wvl_final + 1),
    #         np.linspace(-1, 1, mp.array_size[0]), np.linspace(-1, 1, mp.array_size[1])]
    # bins = [mp.array_size[0]] * 4
    bins = [np.linspace(photonlist[:, 0].min(), photonlist[:, 0].max(), sp.numframes + 1),
            np.linspace(photonlist[:, 1].min(), photonlist[:, 1].max(), ap.n_wvl_final + 1),
            np.linspace(-200, 200, 151),
            np.linspace(-200, 200, 151)]

    tess, _ = np.histogramdd(photonlist, bins=bins)

    return tess

def get_angle_list(tess):
    angle_list = -np.linspace(0, sp.numframes * sp.sample_time * config['data']['rot_rate']/60, tess.shape[0])
    return angle_list

def derot_tess(tess):
    derot = np.sum(cube_derotate(np.sum(tess, axis=1), get_angle_list(tess)), axis=0)
    return derot


