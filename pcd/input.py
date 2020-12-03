import os
import shutil
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import random
import h5py
from vip_hci.medsub import medsub_source
import pandas as pd
pd.options.display.max_columns = None
from scipy.signal import convolve2d
import pickle

from medis.telescope import Telescope
from medis.MKIDS import Camera
from medis.utils import dprint
from medis.plot_tools import grid
from medis.distribution import planck, Distribution

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config, convert_astro, convert_inputfiles
import utils
from utils import confusion_matrix, get_bin_measures


class MedisObs():
    """ Gets the photon lists from MEDIS """
    def __init__(self, name, astro, debug=False):
        """ astro tuple containing contrast, lod and spectra tuple"""
        iop.update_testname(str(name))
        iop.photonlist = iop.photonlist.split('.')[0]+'.pkl'
        self.astro = astro
        print(f"contrast: {np.log10(astro[0])}, loc: {astro[1]*10}, spec temp: {astro[2]}")
        self.numobj = len(ap.contrast) + 1
        if os.path.exists(iop.photonlist):
            with open(iop.photonlist, 'rb') as handle:
                self.photons, self.photlist_astro = pickle.load(handle)

        else:
            self.medis_cache = iop.testdir

            ap.contrast = [astro[0]]
            ap.companion_xy = [astro[1]]
            # ap.spectra = [[spec] for spec in astro[2]]
            ap.spectra = astro[2]

            if ap.contrast == [0.0]:
                ap.contrast = []
                ap.companion = False
            else:
                ap.companion = True

            # instantiate but don't generate data yet
            tel = Telescope(usesave=sp.save_to_disk)
            cam = Camera(usesave=False, product='photons')

            self.photons = []
            for o in range(self.numobj):
                if tel.num_chunks == 1:
                    fields = tel()['fields']
                    object_fields = fields[:,:,:,o][:,:,:,np.newaxis]
                    photons = cam(fields=object_fields)['photons']
                else:
                    photons = np.empty((4,0))
                    tel.chunk_span = np.arange(0, sp.numframes+1, sp.checkpointing)
                    for ichunk in range(int(np.ceil(tel.num_chunks))):
                        fields = tel()['fields']
                        object_fields = fields[:, :, :, o][:, :, :, np.newaxis]
                        photons = np.hstack((photons, cam(fields=object_fields, abs_step=tel.chunk_span[ichunk])['photons']))

                tel.chunk_ind = 0
                tel.fields_exists = True  # this is defined during Telescope.__init__ so redefine here once fields is made

                if config['data']['time_diff']:
                    stem = cam.arange_into_stem(photons.T, (cam.array_size[1], cam.array_size[0]))
                    stem = list(map(list, zip(*stem)))
                    stem = cam.calc_arrival_diff(stem)
                    photons = cam.ungroup(stem)

                self.photons.append(photons.T)

            self.photlist_astro = self.astro
            with open(iop.photonlist, 'wb') as handle:
                pickle.dump((self.photons, self.photlist_astro), handle, protocol=pickle.HIGHEST_PROTOCOL)

        if debug:
            self.display_2d_hists()

    def adjust_companion(self):
        # brightness change
        ratio = self.astro[0]/self.photlist_astro[0] # 10. ** config['data']['contrasts'][0]
        print(ratio, self.photlist_astro[0], self.astro[0])
        assert ratio <= 1
        planet_inds = range(len(self.photons[1]))
        del_planet_inds = np.sort(random.sample(list(planet_inds), int(len(planet_inds) * (1-ratio))))
        if len(del_planet_inds)>0:
            self.photons[1] = np.delete(self.photons[1], del_planet_inds, axis=0)

        # location changes
        planet_photons = self.photons[1]
        angles = np.deg2rad(planet_photons[:,0] * config['data']['rot_rate']/60)
        yc, xc = np.mean(planet_photons[:,2]), np.mean(planet_photons[:,3])
        rad_offset = self.astro[1] * 10
        planet_photons[:,3] += -xc + rad_offset[1]
        planet_photons[:,2] += -yc + rad_offset[0]
        x_rot = planet_photons[:,3] * np.cos(angles) - planet_photons[:,2] * np.sin(angles)
        y_rot = planet_photons[:,3] * np.sin(angles) + planet_photons[:,2] * np.cos(angles)
        planet_photons[:, 3], planet_photons[:,2] = x_rot + mp.array_size[0]/2, y_rot + mp.array_size[0]/2
        for a in [planet_photons[:, 3], planet_photons[:, 2]]:
            a[a<0] = 0
            a[a>=mp.array_size[0]] = mp.array_size[0]-1

        # spectrum changes
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_final)
        spectrum = planck(self.astro[2][1], wsamples)
        spectrum /= np.sum(spectrum)
        interp = True if len(planet_photons) > 1 else False
        dist = Distribution(spectrum, interpolation=interp)

        phot_spec = dist(len(planet_photons))[0]
        planet_photons[:, 1] = phot_spec*mp.array_size[0]/np.max(phot_spec) - mp.array_size[0]

        self.photons[1] = planet_photons

    def display_2d_hists(self):
        fig, axes = utils.init_grid(rows=self.numobj, cols=4, figsize=(16,4*self.numobj))

        if config['mec']['dithered']:
            bins = [np.linspace(0, sp.sample_time * sp.numframes, 50), np.linspace(-mp.array_size[0], 0, 50),
                    np.linspace(self.photons[0][:,2].min(), self.photons[0][:,2].max(), mp.array_size[0]),
                    np.linspace(self.photons[0][:,3].min(), self.photons[0][:,3].max(), mp.array_size[0])]
        else:
            bins = [np.linspace(0, sp.sample_time * sp.numframes, 50), np.linspace(-mp.array_size[0], 0, 50), range(mp.array_size[0]),
                    range(mp.array_size[1])]

        # if config['data']['trans_polar']:
        #     bins[2] = np.linspace(0, np.sqrt(((mp.array_size[0]/2)**2) * 2), mp.array_size[0])
        #     bins[3] = np.linspace(-np.pi,np.pi, mp.array_size[1])

        coord = 'tpxy'
        for o in range(self.numobj):
            H, _ = np.histogramdd(self.photons[o], bins=bins)

            for p, pair in enumerate([['y','x'], ['x','p'], ['x','t'], ['p','t']]):
                inds = coord.find(pair[0]), coord.find(pair[1])
                sumaxis = tuple(np.delete(range(len(coord)), inds))
                image = np.sum(H, axis=sumaxis)
                if pair in [['x','p'], ['x','t']]:
                    image = image.T
                    inds = inds[1], inds[0]
                axes[o,p].imshow(image, aspect='auto', origin='lower', norm=LogNorm(),
                                 extent=[bins[inds[0]][0],bins[inds[0]][-1],bins[inds[1]][0],bins[inds[1]][-1]])

        plt.show(block=True)

class Reform():
    def __init__(self, photons, outfile, train_type='train', aug_ind=0, debug=False, dithered=False):
        self.photons = photons #[self.normalise_photons(photons[o]) for o in range(config['classes'])]
        self.outfile = outfile
        self.prefix = outfile.split('.')[0]
        self.train_type = train_type
        self.aug_ind = aug_ind
        self.debug = debug
        self.dithered = dithered

        self.num_point = config['num_point']
        # self.test_frac = config['data']['test_frac']
        self.dimensions = config['dimensions']
        assert self.dimensions in [3,4]
        self.num_classes = len(photons)  # not config['classes'] because sometimes there are only photons for one class

    def aggregate_photons(self):
        self.all_photons = np.empty((0, self.dimensions))  # photonlist with both types of photon
        self.all_pids = np.empty((0, 1))  # associated photon labels

        for o in range(self.num_classes):
            self.all_photons = np.concatenate((self.all_photons, self.photons[o]), axis=0)
            self.all_pids = np.concatenate((self.all_pids, np.ones_like((self.photons[o][:, [0]])) * int(o>0)), axis=0)

    def sort_photons(self):
        # sort by time so the planet photons slot amongst the star photons at the appropriate point
        time_sort = np.argsort(self.all_photons[:, 0]).astype(int)

        self.all_photons = self.all_photons[time_sort]
        self.all_pids = self.all_pids[time_sort]

    def normalise_photons(self, use_bounds=True):
        # normalise photons
        if use_bounds:
            bounds = np.array([[0, sp.sample_time * sp.numframes],
                               mp.wavecal_coeffs[0] * ap.wvl_range + mp.wavecal_coeffs[1],  # [-116, 0], ap.wvl_range = np.array([800, mp.array_size[0]0]), mp.wavecal_coeffs = [1. / 6, -250]
                               [0, mp.array_size[0]],
                               [0, mp.array_size[1]]])
            # if config['data']['trans_polar']:
            #     bounds[2] = [0, np.sqrt(((mp.array_size[0]/2)**2) * 2)]
            #     bounds[3] = [-np.pi, np.pi]
            self.all_photons -= np.mean(bounds, axis=1)
            self.all_photons /= np.max(self.all_photons, axis=0)
        else:
            self.all_photons -= (np.min(self.all_photons, axis=0) + np.max(self.all_photons, axis=0))/2  # np.mean(self.all_photons, axis=0)
            self.all_photons /= (2*np.std(self.all_photons, axis=0))
        self.normalised = True

class NnReform(Reform):
    """ Creates the input data in the NN input format """
    def __init__(self, photons, outfile, astro, train_type='train', aug_ind=0, debug=False, dithered=False):
        super().__init__(photons, outfile, train_type, aug_ind, debug, dithered)

        self.chunked_photons = []
        self.chunked_pids = []
        self.labels = []
        self.data = []
        self.pids = []
        self.astro = astro

        self.normalised = False

    def process_photons(self):
        self.aggregate_photons()
        self.sort_photons()
        self.normalise_photons(use_bounds=not self.dithered)
        self.chunk_photons()

    def chunk_photons(self):
        # remove residual photons that won't fit into a input cube for the network
        total_photons = sum([len(self.photons[i]) for i in range(self.num_classes)])
        # cut = int(total_photons % self.num_point)
        # dprint(total_photons, cut)
        # rand_cut = random.sample(range(total_photons), cut)
        # red_photons = np.delete(self.all_photons, rand_cut, axis=0)
        # red_pids = np.delete(self.all_pids, rand_cut, axis=0)

        if total_photons > self.num_point:
            rand_keep = random.sample(range(total_photons), self.num_point)
            total_photons = self.num_point
        else:
            rand_keep = np.arange(total_photons)
        self.chunked_photons = self.all_photons[rand_keep]
        self.chunked_pids = self.all_pids[rand_keep]

        self.chunked_photons = self.chunked_photons.reshape(1, total_photons, self.dimensions, order='F')  # selects them throughout the obss
        self.chunked_pids = self.chunked_pids.reshape(1, total_photons, 1, order='F')

    def save_class(self):

        num_input = len(self.chunked_photons)  # 16

        reorder = np.apply_along_axis(np.random.permutation, 1,
                                      np.ones((num_input, self.chunked_photons.shape[1])) * np.arange(self.chunked_photons.shape[1])).astype(np.int)

        self.data = np.array([self.chunked_photons[o, order] for o, order in enumerate(reorder)])
        if config['task'] == 'part_seg':
            self.labels = np.ones((num_input), dtype=int) #* self.label
            self.pids = np.array([self.chunked_pids[o, order] for o, order in enumerate(reorder)])[:, :, 0]
        else:
            self.labels = np.array([self.chunked_pids[o, order] for o, order in enumerate(reorder)])[:, :, 0]

        if config['pointnet_version'] == 2:
            if self.train_type == 'train':
                self.smpw = [self.chunked_pids.size/(self.chunked_pids == o).sum() for o in range(self.num_classes)]

                # labelweights, _ = np.histogram(self.chunked_pids, range(self.num_classes+1))
                # labelweights = labelweights.astype(np.float32)
                # labelweights = labelweights/np.sum(labelweights)
                # self.smpw = 1/np.log(1.2+labelweights)
                # self.smpw = labelweights
                dprint(self.smpw)
            else:
                self.smpw = np.ones((self.num_classes))

        # self.data = self.data[:, :, [1, 3, 0, 2]]

        if self.debug:
            self.display_2d_hists()

        with h5py.File(self.outfile, 'w') as hf:

            hf.create_dataset('data', data=self.data)
            hf.create_dataset('label', data=self.labels)
            if config['task'] == 'part_seg':
                hf.create_dataset('pid', data=self.pids)
            if config['pointnet_version'] == 2:
                hf.create_dataset('smpw', data=self.smpw)
            hf.attrs[u'contrast'] = self.astro[0]
            hf.attrs[u'loc'] = self.astro[1] * 10
            hf.attrs[u'spec'] = self.astro[2][1]

    def adjust_companion(self, astro):
        # brightness change
        ratio = astro[0] / 10. ** config['data']['contrasts'][0]
        assert ratio <= 1
        planet_inds = np.where(self.chunked_pids[0,:,0])[0]
        # del_planet_inds = np.sort(np.random.choice(planet_inds, int(len(planet_inds) * (1-ratio))))
        del_planet_inds = np.sort(random.sample(list(planet_inds), int(len(planet_inds) * (1-ratio))))
        self.chunked_pids = np.delete(self.chunked_pids, del_planet_inds, axis=1)
        self.chunked_photons = np.delete(self.chunked_photons, del_planet_inds, axis=1)

        # location changes
        planet_inds = np.where(self.chunked_pids[0, :, 0])[0]
        planet_photons = self.chunked_photons[0,planet_inds]
        angles = np.deg2rad((planet_photons[:,0]+1) * 0.5 * sp.numframes * sp.sample_time * config['data']['rot_rate']/60)
        yc, xc = np.mean(planet_photons[:,2]), np.mean(planet_photons[:,3])
        rad_offset = astro[1] * 2 * 10 / mp.array_size[0]
        planet_photons[:,3] += -xc + rad_offset[1]
        planet_photons[:,2] += -yc + rad_offset[0]
        x_rot = planet_photons[:,3] * np.cos(angles) - planet_photons[:,2] * np.sin(angles)
        y_rot = planet_photons[:,3] * np.sin(angles) + planet_photons[:,2] * np.cos(angles)
        planet_photons[:, 3], planet_photons[:,2] = x_rot, y_rot

        # spectrum changes
        wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_final)
        spectrum = planck(astro[2][1], wsamples)
        spectrum /= np.sum(spectrum)
        dist = Distribution(spectrum)
        phot_spec = dist(len(planet_photons))
        planet_photons[:, 1] = phot_spec*2/np.max(phot_spec) - 1

        self.chunked_photons[0, planet_inds] = planet_photons

    def display_chunk_cloud(self, downsamp=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange']

        for t in range(10):
            print(t)
            for i, c in enumerate(colors):
                print(i, c, len(self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 0][::downsamp]))
                ax.scatter(self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 0][::downsamp],
                           self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 1][::downsamp],
                           self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 2][::downsamp], c=c,
                           marker='.')  # , marker=pids[0])
        plt.show(block=True)

    def display_2d_hists(self, ind=None):
        fig, axes = utils.init_grid(rows=self.num_classes, cols=config['dimensions'], figsize=(16,4*self.num_classes))
        # fig, axes = utils.init_grid(rows=self.num_classes, cols=4)
        fig.suptitle(f'{ind}', fontsize=16)
        plt.tight_layout()

        if not self.normalised:
            bins = [np.linspace(0, sp.sample_time * sp.numframes, 50), np.linspace(-120, 0, 50), range(mp.array_size[0]),
                    range(mp.array_size[1])]
        else:
            radius = 1.5
            bins = [np.linspace(-radius,radius,50), np.linspace(-radius,radius,50), 
                    np.linspace(-radius,radius,mp.array_size[0]), np.linspace(-radius,radius,mp.array_size[0])]

        coord = 'tpxy'

        for o in range(self.num_classes):
            if ind:
                H, _ = np.histogramdd(self.data[ind, (self.labels[ind] == o)], bins=bins)
            else:
                H, _ = np.histogramdd(self.data[self.labels==o], bins=bins)

            for p, pair in enumerate([['x','y'], ['x','p'], ['x','t'], ['p','t']]):
                inds = coord.find(pair[0]), coord.find(pair[1])
                sumaxis = tuple(np.delete(range(len(coord)), inds))
                image = np.sum(H, axis=sumaxis)
                if pair in [['x','p'], ['x','t']]:
                    image = image.T
                    inds = inds[1], inds[0]
                axes[o,p].imshow(image, norm=LogNorm(), aspect='auto',
                                 extent=[bins[inds[0]][0],bins[inds[0]][-1],bins[inds[1]][0],bins[inds[1]][-1]])

        plt.show(block=True)

class DtReform(Reform):
    def __init__(self, photons, outfile, astro, train_type='train', aug_ind=0, debug=False, dithered=False):
        super().__init__(photons, outfile, train_type, aug_ind, debug, dithered)

        self.aggregate_photons()
        self.sort_photons()
        self.astro= astro

        self.df = pd.DataFrame({'time':self.all_photons[:,0],
                                'wave':self.all_photons[:,1],
                                'x':self.all_photons[:,2].astype(int),
                                'y':self.all_photons[:,3].astype(int)})

        self.df['exo'] = self.all_pids[:,0]

        self.id_bins = np.arange(mp.array_size[0] * mp.array_size[1])
        beammap = self.id_bins.reshape(mp.array_size)
        self.id_bins = np.concatenate((self.id_bins, [self.id_bins[-1] + 1]), axis=0) - 0.5

        self.df['res_id'] = beammap[self.df['x'].values.astype(int), self.df['y'].values.astype(int)]
        self.df = self.df.sort_values(['res_id', 'time'])

        dt = self.df['time'].values - np.roll(self.df['time'].values,1,0)
        trans = np.roll(self.df['res_id'].values,1,0) - self.df['res_id'].values != 0
        dt[trans] = dt[np.roll(trans,1,0)]  #np.nan

        self.df['dtime'] = dt

        exo_ind = self.df['exo']==1
        exo_res, amounts = np.unique(self.df.loc[exo_ind]['res_id'], return_counts=True)

        bright_exo_res = exo_res[amounts > np.max(amounts)/10]
        bright_res_inds = self.df[self.df['res_id'].isin(bright_exo_res)].index  # indicies of photons that are on those res_ids
        self.df.loc[bright_res_inds, 'exo'] = 1

    def binfree_ADI(self, dists=1, N=50, fwhm=10, plot_dict={}):
        """ todo reformat to match this format https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-map-attributeerror-cant-pickle-local-object/52283968 """

        self.df['bfadi_exo'] = np.zeros(len(self.df))

        if plot_dict['in_map']:
            I, _ = np.histogram(self.df['res_id'], self.id_bins)
            plt.figure(figsize=(12,12))
            plt.imshow(I.reshape(mp.array_size), norm=LogNorm(), origin='lower')

        # rads = np.arange(np.round((np.min(config['data']['lods'])-0.5) * fwhm),
        #                  np.round((np.max(config['data']['lods'])+0.5) * fwhm),
        #                  dists)
        rads = np.arange(55, 70, dists)
        for ir, rad in enumerate(rads):
            if rad % 10 == 0:
                print(f'locating planets at rad {rad}')
            centered_coords = utils.find_coords(rad, dists)
            annuli_coords = np.array([centered_coords[i] + mp.array_size[0]//2 for i in range(2)]).astype(int)

            all_times = np.empty(0)
            all_dts = np.empty(0)
            all_ids = np.empty(0)
            all_exo = np.empty(0)
            for ic, c in enumerate(annuli_coords.T):
                df = self.df[(self.df['x'] == c[0]) & (self.df['y'] == c[1])]
                dts = df['dtime'].values
                cc = centered_coords.T[ic]
                theta = np.rad2deg(np.arctan2(cc[1], cc[0]))
                offset = theta / (config['data']['rot_rate']/60)
                all_times = np.append(all_times, df['time'].values + offset)
                all_dts = np.append(all_dts, dts)
                all_ids = np.append(all_ids, df.index.values)
                all_exo = np.append(all_exo, df['exo'].values)

            sort_ind = np.argsort(all_times)
            all_times = all_times[sort_ind]
            all_dts = all_dts[sort_ind]
            all_ids = all_ids[sort_ind]
            all_exo = all_exo[sort_ind]

            # todo run this on the timestreams
            # https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
            # from scipy.spatial.distance import pdist, squareform
            #
            # def rec_plot(s, eps=0.1, steps=10):
            #     d = pdist(s[:, None])
            #     d = np.floor(d / eps)
            #     d[d > steps] = steps
            #     Z = squareform(d)
            #     return Z

            # sma = np.convolve(all_dts, np.ones((N,)) / N, mode='same')
            sma = convolve2d(all_dts[:,np.newaxis], np.ones((N,))[:,np.newaxis] / N, mode='same', boundary='wrap')
            bfadi_exo = sma < np.median(all_dts)

            self.df.loc[all_ids, 'bfadi_exo'] = bfadi_exo

            if plot_dict['Annulus stream']:
                exo_loc = np.sqrt(np.sum(self.astro[1]**2)) * 10
                print(exo_loc, rads[ir-1], rads[ir], exo_loc >= rads[ir-1] and exo_loc < rads[ir])
                if exo_loc >= rads[ir-1] and exo_loc < rads[ir]:
                    plt.figure()
                    plt.title(f'{rad} annulus combined time stream')
                    plt.plot(all_times, all_dts)
                    plt.plot(all_times, sma)
                    plt.axhline(np.median(all_dts))
                    plt.plot(all_times, bfadi_exo)
                    plt.plot(all_times, all_exo)
                    plt.tight_layout()
                    plt.show()

            if plot_dict['Each pix stream'] == rad:
                fig = plt.figure(figsize=(12,12))
                for ic, c in enumerate(annuli_coords.T[:50]):
                    df = self.df[(self.df['x'] == c[0]) & (self.df['y'] == c[1])]
                    dts = df['dtime'].values
                    bfadi_exo = df['bfadi_exo'].values
                    cc = centered_coords.T[ic]
                    theta = np.rad2deg(np.arctan2(cc[1], cc[0]))
                    offset = theta / config['data']['rot_rate']
                    ax = fig.add_subplot(5, 10, ic + 1)
                    ax.plot(df['time'].values + offset, dts)
                    ax.plot(df['time'].values + offset, bfadi_exo)
                plt.tight_layout()
                plt.show()

        true_pos, false_neg, \
        false_pos, true_neg = get_bin_measures(self.df['exo'].values.astype(int),
                                                       self.df['bfadi_exo'].values.astype(int),
                                                       sum=True)
        print(confusion_matrix(false_neg, true_pos, true_neg, false_pos, true_neg + false_pos, true_pos + false_neg))

        # testdf = self.df.sort_values(['x', 'y'])
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(testdf.loc[testdf['exo'] != testdf['bfadi_exo']][['x', 'y', 'exo', 'bfadi_exo']])

        planet_df = self.df.loc[self.df['bfadi_exo'] == 1]

        if plot_dict['out_map']:
            bins = [np.arange(mp.array_size[0])] * 2
            metric_inds = get_bin_measures(self.df['exo'].values.astype(int),
                                                    self.df['bfadi_exo'].values.astype(int),
                                                    sum=False)
            titles = ['true_pos', 'false_neg', 'false_pos', 'true_neg']
            fig = plt.figure()
            for i,m in enumerate(metric_inds):
                image, _, _ = np.histogram2d(self.df.loc[m]['x'], self.df.loc[m]['y'], bins)
                image[image==0] = 0.5
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(image, norm=LogNorm(), origin='lower')
                ax.set_title(titles[i])
            plt.tight_layout()
            plt.show()

        photons = [planet_df.loc[planet_df['exo']==i][['time', 'wave', 'x', 'y']].values for i in range(config['classes'])]
        # self.df = self.df.drop(self.df.loc[self.df['bfadi_exo'] == 0].index)
        # photons = [self.df.loc[self.df['exo'] == i][['time', 'wave', 'x', 'y']].values for i in range(config['classes'])]

        return photons


class MedisParams():
    """ Infers the sequence of parameters to pass to each MedisObs """
    def __init__(self, config=None):

        self.contrasts = np.power(np.ones((config['data']['num_indata']))*10, config['data']['contrasts'])
        if config['data']['null_frac'] > 0:
            self.contrasts[::int(1 / config['data']['null_frac'])] = 0
        invalid_contrast = np.array(config['data']['contrasts']) > 0
        self.contrasts[invalid_contrast] = 0
        maxcont = np.argmax(self.contrasts)
        self.contrasts = np.append(self.contrasts[maxcont], np.delete(self.contrasts, maxcont))
        disp = config['data']['lods']
        angle = config['data']['angles']
        self.lods = (np.array([np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))])*disp).T

        self.spectra = [(config['data']['star_spectra'], p_spec) for p_spec in config['data']['planet_spectra']]
        dprint(self.contrasts)

    def __call__(self, ix, *args, **kwargs):
        return (self.contrasts[ix], self.lods[ix], self.spectra[ix])

def load_h5(h5_filename, full_output=False):
    print(h5_filename)
    with h5py.File(h5_filename, 'r') as f:
        if config['data']['degrade_factor'] != 1:
            subsample = random.sample(range(config['num_point']), int(config['num_point'] / config['data']['degrade_factor']))
        else:
            subsample = ...
        data = f['data'][:][:,subsample]
        label = f['label'][:][:,subsample]
        smpw = f['smpw'][:]
        print(config['data']['degrade_factor'], data.shape)
        try:
            contrast = f.attrs['contrast']
            loc = f.attrs['loc']
            spec = f.attrs['spec']
            print(f"contrast: {np.log10(contrast)}, loc: {loc}, spec temp: {spec}")
        except KeyError:
            print('no astro info known')

    if full_output:
        return data, label, smpw, {'contrast': contrast, 'loc': loc, 'spec': spec}
    else:
        return (data, label, smpw)

def load_dataset(in_file, shuffle=False):
    assert os.path.isfile(in_file), f'[error] {in_file} dataset path not found'

    print(f'loading {in_file}')
    in_data, in_label, class_weights, astro_dict = load_h5(in_file, full_output=True)

    if shuffle:
        raise NotImplementedError

    return in_data, in_label, astro_dict

def make_input(config):
    mp = MedisParams(config)

    # get info on each photoncloud
    outfiles = np.append(config['trainfiles'], config['testfiles'])
    debugs = [False] * config['data']['num_indata']
    # debugs[0] = False
    train_types = ['train'] * config['data']['num_indata']
    num_test = config['data']['num_indata'] * config['data']['test_frac']
    num_test = int(num_test)
    if num_test > 0: train_types[-num_test:] = ['test']*num_test
    aug_inds = np.arange(config['data']['num_indata'])
    aug_inds[::config['data']['aug_ratio']+1] = 0  # eg [0,1,2,3,0,5,6,7,0,9,10,11,...] when aug_ratio == 3

    for i, outfile, train_type, aug_ind in zip(range(config['data']['num_indata']), outfiles, train_types, aug_inds):
        print(f'Creating outfile {outfile} ...')
        if os.path.exists(outfile):
            print('Already exists')
        else:
            astro = mp(i)
            med_ind = np.arange(config['data']['num_indata'])[i]//(config['data']['aug_ratio']+1)
            obs = MedisObs(f'{med_ind}', astro, debug=False)
            obs.adjust_companion()
            photons = obs.photons

            if config['pre_adi']:
                r = DtReform(photons, outfile, astro=astro, train_type=train_type, aug_ind=aug_ind, debug=debugs[i])
                photons = r.binfree_ADI(plot_dict={'in_map':False, 'Annulus stream': True, 'Each pix stream': False, 'out_map': True})

            r = NnReform(photons, outfile, train_type=train_type, aug_ind=aug_ind, debug=debugs[i], astro=astro)
            r.process_photons()
            r.save_class()

    workingdir_config = config['working_dir'] + 'config.yml'
    repo_config = os.path.join(os.path.dirname(__file__), 'config/config.yml')
    if not os.path.exists(workingdir_config):
        shutil.copyfile(repo_config, workingdir_config)

def make_eval():
    """ Creates a single input file with what you need for hyper-params tests for example """
    config['data']['num_indata'] = 9
    config['data']['test_frac'] = 1
    config['data']['contrasts'] = '(-4,-2.05)'
    config['data']['angles'] = '(0,360)'
    config['data']['lods'] = '(2,7)'
    config['trainfiles'] = []
    config['testfiles'] = [os.path.join(config['working_dir'], f'testfile_{id}.h5') for id in
                           range(config['data']['num_indata'])]
    convert_astro(config)
    make_input(config)


if __name__ == "__main__":
    # make_input(config)
    make_eval()
