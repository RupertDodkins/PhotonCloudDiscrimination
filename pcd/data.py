import os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import random
import h5py

import medis.get_photon_data as gpd
import medis.save_photon_data as spd
import medis.Detector.pipeline as pipe
from medis.Utils.misc import dprint

from config.medis_params import sp, ap, tp, iop, mp
from config.config import config
import utils

class Obsfile():
    """ Gets the photon lists from MEDIS """
    def __init__(self, name, contrast, lods, debug=False):
        iop.set_testdir(str(name))
        ap.contrast = contrast
        ap.lods = lods
        self.numobj = config['data']['num_planets']+1

        if not os.path.exists(iop.fields):
            gpd.run_medis()

        self.photons = pipe.read_obs()
        # if config['debug']:
        if debug:
        # #     # self.display_raw_image()
        # #     # self.display_raw_cloud()
            self.display_2d_hists()
            # self.plot_stats()
        dprint(len(self.photons))

    def log_params(self):
        """ Log the MEDIS parameters for reference """
        raise NotImplemented

    def plot_stats(self):
        rad = 4
        starcenter = mp.array_size//2
        planetcenter1 = [50, 75]
        speckcenter = [50, 75]
        centers = [starcenter, speckcenter, planetcenter1]
        objs = ['star', 'speckle', 'planet']
        fields = [0,0,1]

        fig, axes = utils.init_grid(rows=3, cols=3)
        for o, (center, f) in enumerate(zip(centers, fields)):
            objbounds = [center[0]-rad, center[0]+rad, center[1]-rad, center[1]+rad]
            print(objbounds)
            locs = np.all((self.photons[f][:, 2] >= objbounds[0],
                           self.photons[f][:, 2] <= objbounds[1],
                           self.photons[f][:, 3] >= objbounds[2],
                           self.photons[f][:, 3] <= objbounds[3]), axis=0)

            inten = np.histogram(self.photons[f][locs,0], bins=2500)[0]
            axes[0,o].plot(self.photons[f][locs,0])
            axes[0,o].set_title(objs[o])
            axes[1,o].plot(inten)
            axes[2,o].hist(inten, bins=50)

        plt.show(block=True)

    def display_2d_hists(self):
        fig, axes = utils.init_grid(rows=self.numobj, cols=4)

        bins = [np.linspace(0, ap.sample_time * ap.numframes, 50), np.linspace(-120, 0, 50), range(mp.array_size[0]),
                range(mp.array_size[1])]

        coord = 'tpxy'
        for o in range(self.numobj):
            H, _ = np.histogramdd(self.photons[o], bins=bins)

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

    def display_raw_cloud(self, downsamp=10000):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange']
        for i, c in enumerate(colors[:self.numobj]):
            ax.scatter(self.photons[i][:,1][::downsamp], self.photons[i][:,2][::downsamp],
                       self.photons[i][:,3][::downsamp], c=c, marker='.')  # , marker=pids[0])
        plt.show(block=True)

    def display_raw_image(self):
        for o in range(self.numobj):
            fig = plt.figure()
            bins = [np.linspace(0, ap.sample_time * ap.numframes, 3), np.linspace(-90, 0, 4), range(mp.array_size[0]),
                    range(mp.array_size[1])]
            nrows, ncols = len(bins[0])-1, len(bins[1])-1
            gs = gridspec.GridSpec(nrows, ncols)
            for r in range(nrows):
                for c in range(ncols):
                    fig.add_subplot(gs[r, c])
            axes = np.array(fig.axes).reshape(nrows, ncols)

            H, _ = np.histogramdd(self.photons[o], bins=bins)
            print(nrows, ncols, H.shape)
            for r in range(nrows):
                for c in range(ncols):
                    axes[r,c].imshow(H[r,c], norm=LogNorm())
        plt.show()

class Class():
    """ Creates the input data in the NN input format """
    def __init__(self, photons, outfile, debug=False):
        self.photons = photons
        self.outfile = outfile
        self.debug = debug

        self.num_point = config['num_point']
        self.test_frac = config['test_frac']
        self.dimensions = config['dimensions']
        assert self.dimensions in [3,4]

        self.chunked_photons = []
        self.chunked_pids = []
        self.labels = []
        self.data = []
        self.pids = []

    def chunk_photons(self):
        all_photons = np.empty((0, self.dimensions))  # photonlist with both types of photon
        all_pids = np.empty((0, 1))  # associated photon labels
        total_photons = sum([len(self.photons[i]) for i in range(config['classes'])])

        for o in range(config['classes']):
            # dprint((all_photons.shape, self.photons[o][:, [0, 2, 3]].shape))
            if self.dimensions == 3:
                all_photons = np.concatenate((all_photons, self.photons[o][:, [0, 2, 3]]), axis=0)
            else:
                all_photons = np.concatenate((all_photons, self.photons[o]), axis=0)
            all_pids = np.concatenate((all_pids, np.ones_like((self.photons[o][:, [0]])) * int(o>0)), axis=0)

        # sort by time so the planet photons slot amongst the star photons at the appropriate point
        time_sort = np.argsort(all_photons[:, 0]).astype(int)

        all_photons = all_photons[time_sort]
        all_pids = all_pids[time_sort]

        # remove residual photons that won't fit into a input cube for the network
        cut = int(total_photons % self.num_point)
        # dprint(cut)
        rand_cut = random.sample(range(total_photons), cut)
        red_photons = np.delete(all_photons, rand_cut, axis=0)
        red_pids = np.delete(all_pids, rand_cut, axis=0)

        # raster the list so that every self.num_point start a new input cube
        self.chunked_photons = red_photons.reshape(-1, self.num_point, self.dimensions)
        self.chunked_pids = red_pids.reshape(-1, self.num_point, 1)

    def save_class(self):
        num_input = len(self.chunked_photons)  # 16

        reorder = np.apply_along_axis(np.random.permutation, 1,
                                      np.ones((num_input, self.num_point)) * np.arange(self.num_point)).astype(np.int)

        self.data = np.array([self.chunked_photons[o, order] for o, order in enumerate(reorder)])
        if config['task'] == 'part_seg':
            self.labels = np.ones((num_input), dtype=int) #* self.label
            self.pids = np.array([self.chunked_pids[o, order] for o, order in enumerate(reorder)])[:, :, 0]
        else:
            self.labels = np.array([self.chunked_pids[o, order] for o, order in enumerate(reorder)])[:, :, 0]

        # if config['debug']:
        if self.debug:
            # self.display_chunk_cloud()
            self.display_2d_hists()

        with h5py.File(self.outfile, 'w') as hf:
            # hf.create_dataset('data', data=self.data[:-int(self.test_frac * num_input)])
            hf.create_dataset('data', data=self.data)
            hf.create_dataset('label', data=self.labels)
            if config['task'] == 'part_seg':
                hf.create_dataset('pid', data=self.pids)

        # with h5py.File(self.testfile, 'w') as hf:
        #     hf.create_dataset('data', data=self.data[-int(self.test_frac * num_input):])
        #     hf.create_dataset('label', data=self.labels[-int(self.test_frac * num_input):])
        #     if config['task'] == 'part_seg':
        #         hf.create_dataset('pid', data=self.pids[-int(self.test_frac * num_input):])

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
        fig, axes = utils.init_grid(rows=config['classes'], cols=config['dimensions'])
        # fig, axes = utils.init_grid(rows=config['classes'], cols=4)
        fig.suptitle(f'{ind}', fontsize=16)
        plt.tight_layout()

        bins = [np.linspace(0, ap.sample_time * ap.numframes, 50), np.linspace(-120, 0, 50), range(mp.array_size[0]),
                range(mp.array_size[1])]

        coord = 'tpxy'

        for o in range(config['classes']):
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

class Data():
    """ Infers the sequence of parameters to pass to each Obsfile """
    def __init__(self, config=None):
        self.numobj = config['data']['num_planets']+1
        self.contrasts, self.lods = self.observation_params()
        # self.trainfiles, self.testfiles = self.get_filenames()

    def observation_params(self):
        # numplanets = config['max_planets']
        contrasts = np.power(np.ones((config['data']['num_planets']))*10, config['data']['contrasts'])
        disp = config['data']['lods']
        angle = config['data']['angles']
        lods = (np.array([np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))])*disp).T

        return contrasts, lods

from random import sample
def make_input(config):
    d = Data(config)

    contrast = d.contrasts
    lod = d.lods
    photons = Obsfile(name='0', contrast=contrast, lods=lod, debug=True).photons

    outfiles = np.append(config['trainfiles'], config['testfiles'])
    debugs = [False] * config['data']['num_planets']
    debugs[0] = True
    for label, outfile in zip(range(config['data']['num_planets']),outfiles):
        print(label, outfile)
        class_photons = np.array(photons)[[0,label+1]]

        c = Class(class_photons, outfile, debug=debugs[label])
        c.chunk_photons()
        c.save_class()

if __name__ == "__main__":
    make_input(config)
