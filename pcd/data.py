import os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import random
import h5py

import medis.get_photon_data as gpd
import medis.Detector.pipeline as pipe
from medis.Utils.plot_tools import view_datacube

from config.medis_params import sp, ap, tp, iop, mp
from config.config import config

class Obsfile():
    def __init__(self):
        if not os.path.exists(iop.obs_seq):
            fields = gpd.run_medis()
            tess = (np.abs(fields)**2)[0]
            print(np.sum(tess[:,0], axis=(1,2)))
            print(np.sum(tess[:,1], axis=(1,2)))
            view_datacube(tess[0], logAmp=True)
            view_datacube(tess[:,0], logAmp=True)
        self.photons = pipe.read_obs()


class Data(Obsfile):
    def __init__(self, config=None):
        super(Data, self).__init__()
        self.point_num = config['point_num']
        self.test_frac = 0.2
        # if config is not None:
            # self.trainfile = os.path.join(config['pcd_data'], config['date'], config['data']['trainfile'])
            # self.testfile = os.path.join(config['pcd_data'], config['date'], config['data']['testfile'])
        self.trainfile = config['trainfile']
        self.testfile = config['testfile']
        self.dimensions = config['dimensions']
        assert self.dimensions in [3,4]

        self.chunked_photons = []
        self.chunked_pids = []
        self.labels = []
        self.data = []
        self.pids = []

        print(self.photons[0][:10])

    def chunk_photons(self):
        all_photons = np.empty((0, self.dimensions))  # photonlist with both types of photon
        all_pids = np.empty((0, 1))  # associated photon labels
        total_photons = len(self.photons[0]) + len(self.photons[1])

        for o in range(2):
            # dprint((all_photons.shape, self.photons[o][:, [0, 2, 3]].shape))
            if self.dimensions == 3:
                all_photons = np.concatenate((all_photons, self.photons[o][:, [0, 2, 3]]), axis=0)
            else:
                all_photons = np.concatenate((all_photons, self.photons[o]), axis=0)
            all_pids = np.concatenate((all_pids, np.ones_like((self.photons[o][:, [0]])) * o), axis=0)

        # sort by time so the planet photons slot amongst the star photons at the appropriate point
        time_sort = np.argsort(all_photons[:, 0]).astype(int)

        all_photons = all_photons[time_sort]
        all_pids = all_pids[time_sort]

        # remove residual photons that won't fit into a input cube for the network
        cut = int(total_photons % self.point_num)
        # dprint(cut)
        rand_cut = random.sample(range(total_photons), cut)
        red_photons = np.delete(all_photons, rand_cut, axis=0)
        red_pids = np.delete(all_pids, rand_cut, axis=0)

        # print(all_photons.shape, red_photons.shape, len(rand_cut))

        # raster the list so that every self.point_num start a new input cube
        self.chunked_photons = red_photons.reshape(-1, self.point_num, self.dimensions)
        self.chunked_pids = red_pids.reshape(-1, self.point_num, 1)

        # dprint(self.chunked_photons.shape, self.chunked_pids.shape, np.shape(self.chunked_pids[0] == 0),
        #        np.shape(np.transpose([self.chunked_pids[0] == 0])), np.shape(np.transpose(self.chunked_pids[0] == 0)),
        #        np.shape((self.chunked_pids[0] == 0)[:, 0]))  # , np.shape(self.chunked_photons[0, [self.chunked_pids[0]==0], 0]))
        # dprint(np.shape(self.chunked_photons[0, (self.chunked_pids[0] == 0)[:, 0], 0]))

    def make_train(self):
        num_input = len(self.chunked_photons)  # 16


        reorder = np.apply_along_axis(np.random.permutation, 1,
                                      np.ones((num_input, self.point_num)) * np.arange(self.point_num)).astype(np.int)

        self.labels = np.zeros((num_input), dtype=int)
        self.data = np.array([self.chunked_photons[o, order] for o, order in enumerate(reorder)])
        self.pids = np.array([self.chunked_pids[o, order] for o, order in enumerate(reorder)])[:, :, 0]

        with h5py.File(self.trainfile, 'w') as hf:
            hf.create_dataset('data', data=self.data[:-int(self.test_frac * num_input)])
            hf.create_dataset('label', data=self.labels[:-int(self.test_frac * num_input)])
            hf.create_dataset('pid', data=self.pids[:-int(self.test_frac * num_input)])

        with h5py.File(self.testfile, 'w') as hf:
            hf.create_dataset('data', data=self.data[-int(self.test_frac * num_input):])
            hf.create_dataset('label', data=self.labels[-int(self.test_frac * num_input):])
            hf.create_dataset('pid', data=self.pids[-int(self.test_frac * num_input):])

    def display_chunk_cloud(self, downsamp=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange']
        # bounds = [0,len(photons[0]),data.shape[1]]
        # ax.scatter(data[:, bounds[i]:bounds[i + 1], 0], data[0, bounds[i]:bounds[i + 1], 1],
        #            data[0, bounds[i]:bounds[i + 1], 2], c=c)
        for t in range(10):
            print(t)
            for i, c in enumerate(colors):
                print(i, c, len(self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 0][::downsamp]))
                ax.scatter(self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 0][::downsamp],
                           self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 1][::downsamp],
                           self.chunked_photons[t, (self.chunked_pids[t,:,0] == i), 2][::downsamp], c=c,
                           marker='.')  # , marker=pids[0])
        plt.show(block=True)

    def display_raw_cloud(self, downsamp=1000):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange']
        # bounds = [0,len(photons[0]),data.shape[1]]
        # ax.scatter(data[:, bounds[i]:bounds[i + 1], 0], data[0, bounds[i]:bounds[i + 1], 1],
        #            data[0, bounds[i]:bounds[i + 1], 2], c=c)
        for i, c in enumerate(colors):
            ax.scatter(self.photons[i][:,1][::downsamp], self.photons[i][:,2][::downsamp],
                       self.photons[i][:,3][::downsamp], c=c, marker='.')  # , marker=pids[0])
        plt.show(block=True)

    def display_raw_image(self):
        for o in range(2):
            fig = plt.figure()
            bins = [np.linspace(0, ap.sample_time * ap.numframes, 2), np.linspace(-90, 0, 4), range(mp.array_size[0]),
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

def make_input(config, debug=False):
    debug = config['debug']

    d = Data(config)
    if debug:
        d.display_raw_image()
        d.display_raw_cloud()

    d.chunk_photons()
    if debug:
        d.display_chunk_cloud()

    d.make_train()

if __name__ == "__main__":
    make_input(config)
