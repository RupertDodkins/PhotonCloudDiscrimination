import os
import shutil
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # don't listen to pycharm this is necessary
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import random
import h5py

from medis.telescope import Telescope
from medis.MKIDS import Camera
from medis.utils import dprint
from medis.plot_tools import grid

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config
import utils
from visualization import trans_p2c

class MedisObs():
    """ Gets the photon lists from MEDIS """
    def __init__(self, name, contrast, lods, spectra, debug=False):
        iop.update_testname(str(name))
        self.medis_cache = iop.testdir
        if contrast == [0.0]:
            contrast = []
            ap.companion = False
        else:
            ap.companion = True
        ap.contrast = contrast
        ap.companion_xy = lods
        ap.spectra = spectra

        self.numobj = len(contrast)+1

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
                # photons = photons[[0, 1, 3, 2]]

            if config['data']['trans_polar']:
                photons[2] -= cam.array_size[1]/2
                photons[3] -= cam.array_size[0]/2
                r = np.sqrt(photons[2]**2 + photons[3]**2)
                t =  np.arctan2(photons[3],photons[2])
                photons[-2:] = np.array([r,t])

            self.photons.append(photons.T)

        if debug:
            self.display_2d_hists()

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

        bins = [np.linspace(0, sp.sample_time * sp.numframes, 50), np.linspace(-250, 0, 50), range(mp.array_size[0]),
                range(mp.array_size[1])]

        if config['data']['trans_polar']:
            bins[2] = np.linspace(0, np.sqrt(((mp.array_size[0]/2)**2) * 2), mp.array_size[0])
            bins[3] = np.linspace(-np.pi,np.pi, mp.array_size[1])

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
                axes[o,p].imshow(image, aspect='auto',
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
            bins = [np.linspace(0, sp.sample_time * sp.numframes, 3), np.linspace(-90, 0, 4), range(mp.array_size[0]),
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


class NnReform():
    """ Creates the input data in the NN input format """
    def __init__(self, photons, outfile, train_type='train', aug_ind=0, debug=False, rm_input=None):
        self.photons = photons #[self.normalise_photons(photons[o]) for o in range(config['classes'])]
        self.outfile = outfile
        self.prefix = outfile.split('.')[0]
        self.train_type = train_type
        self.aug_ind = aug_ind
        self.debug = debug
        self.rm_input = rm_input

        self.num_point = config['num_point']
        # self.test_frac = config['data']['test_frac']
        self.dimensions = config['dimensions']
        assert self.dimensions in [3,4]
        self.num_classes = len(photons)  # not config['classes'] because sometimes there are only photons for one class

        self.chunked_photons = []
        self.chunked_pids = []
        self.labels = []
        self.data = []
        self.pids = []

        self.normalised = False

    def process_photons(self):
        self.aggregate_photons()
        self.sort_photons()
        self.normalise_photons()
        self.chunk_photons()

    def aggregate_photons(self):
        self.all_photons = np.empty((0, self.dimensions))  # photonlist with both types of photon
        self.all_pids = np.empty((0, 1))  # associated photon labels

        for o in range(self.num_classes):
            # dprint((self.all_photons.shape, self.photons[o][:, [0, 2, 3]].shape))
            if self.dimensions == 3:
                self.all_photons = np.concatenate((self.all_photons, self.photons[o][:, [0, 2, 3]]), axis=0)
            else:
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
                               # mp.wavecal_coeffs[0] * ap.wvl_range + mp.wavecal_coeffs[1],  # [-116, 0], ap.wvl_range = np.array([800, 1500]), mp.wavecal_coeffs = [1. / 6, -250]
                               [-250, 0],
                               [0, mp.array_size[0]],
                               [0, mp.array_size[1]]])
            if config['data']['trans_polar']:
                bounds[2] = [0, np.sqrt(((mp.array_size[0]/2)**2) * 2)]
                bounds[3] = [-np.pi, np.pi]
            self.all_photons -= np.mean(bounds, axis=1)
            self.all_photons /= np.max(self.all_photons, axis=0)
        else:
            self.all_photons -= np.mean(self.all_photons, axis=0)
            self.all_photons /= np.std(self.all_photons, axis=0)
        self.normalised = True

    def chunk_photons(self):
        # remove residual photons that won't fit into a input cube for the network
        total_photons = sum([len(self.photons[i]) for i in range(self.num_classes)])
        cut = int(total_photons % self.num_point)
        # dprint(cut)
        rand_cut = random.sample(range(total_photons), cut)
        red_photons = np.delete(self.all_photons, rand_cut, axis=0)
        red_pids = np.delete(self.all_pids, rand_cut, axis=0)

        if config['model'] != 'minkowski':
            # raster the list so that every self.num_point start a new input cube
            self.chunked_photons = red_photons.reshape(-1, self.num_point, self.dimensions, order='F')  # selects them throughout the obss
            self.chunked_pids = red_pids.reshape(-1, self.num_point, 1, order='F')
        else:
            self.chunked_photons = red_photons.reshape(1, self.num_point, self.dimensions, order='F')  # selects them throughout the obss
            self.chunked_pids = red_pids.reshape(1, self.num_point, 1, order='F')


    def save_class(self):
        if self.aug_ind:
            self.aug_input()

        num_input = len(self.chunked_photons)  # 16

        reorder = np.apply_along_axis(np.random.permutation, 1,
                                      np.ones((num_input, self.num_point)) * np.arange(self.num_point)).astype(np.int)

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

        if self.rm_input:
            try:
                # shutil.rmtree(self.rm_input)
                dprint(self.rm_input)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    def aug_input(self):
        rot_rate = 360. / (config['data']['aug_ratio'] + 1)
        angle = np.deg2rad(rot_rate * self.aug_ind)
        rot_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        # rot_matrix = np.array(rot_matrix)[:,:,np.newaxis, np.newaxis]

        rotated_photons = self.chunked_photons * 1
        for i in range(len(self.chunked_photons)):
            # self.chunked_photons[i,(self.chunked_pids[i,:,0] == 1)] = np.dot(rot_matrix, self.chunked_photons[i,(self.chunked_pids[i,:,0] == 1)])
            rotated_photons[i,:,[2,3]] = np.dot(rot_matrix, self.chunked_photons[i,:,[2,3]])
            self.chunked_photons[i, (self.chunked_pids[i, :, 0] == 1)] = rotated_photons[i, (self.chunked_pids[i, :, 0] == 1)]
        dprint(rot_rate, self.aug_ind, angle, rot_matrix)

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
        fig, axes = utils.init_grid(rows=self.num_classes, cols=config['dimensions'])
        # fig, axes = utils.init_grid(rows=self.num_classes, cols=4)
        fig.suptitle(f'{ind}', fontsize=16)
        plt.tight_layout()

        if not self.normalised:
            bins = [np.linspace(0, sp.sample_time * sp.numframes, 50), np.linspace(-120, 0, 50), range(mp.array_size[0]),
                    range(mp.array_size[1])]
        else:
            bins = [np.linspace(-1,1,50), np.linspace(-1,1,50), np.linspace(-1,1,150), np.linspace(-1,1,150)]

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


class Data():
    """ Infers the sequence of parameters to pass to each MedisObs """
    def __init__(self, config=None):
        self.numobj = config['data']['num_indata']+1
        self.contrasts, self.lods = self.observation_params()
        # self.trainfiles, self.testfiles = self.get_filenames()

    def observation_params(self):
        # numplanets = config['max_planets']
        contrasts = np.power(np.ones((config['data']['num_indata']))*10, config['data']['contrasts'])
        if config['data']['null_frac'] > 0:
            contrasts[::int(1 / config['data']['null_frac'])] = 0
        # invalid_contrast = np.array(config['data']['contrasts']) == -10
        # contrasts[invalid_contrast] = 0
        disp = config['data']['lods']
        angle = config['data']['angles']
        lods = (np.array([np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))])*disp).T

        return contrasts, lods

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    smpw = f['smpw'][:]
    return (data, label, smpw)

def load_dataset(in_files, shuffle=False):
    for in_file in in_files:
        assert os.path.isfile(in_file), '[error] dataset path not found'

    in_data = np.empty((0, config['num_point'], config['dimensions']))
    in_label = np.empty((0, config['num_point']))

    for in_file in in_files:
        print(f'loading {in_file}')
        file_in_data, file_in_label, class_weights = load_h5(in_file)
        in_data = np.concatenate((in_data, file_in_data), axis=0)
        in_label = np.concatenate((in_label, file_in_label), axis=0)

    if shuffle:
        raise NotImplementedError

    return in_data, in_label

def make_input(config):
    d = Data(config)

    # get info on each photoncloud
    outfiles = np.append(config['trainfiles'], config['testfiles'])
    debugs = [False] * config['data']['num_indata']
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
            if not aug_ind:  # any number > 0
                contrast = [d.contrasts[i]]
                lods = [d.lods[i]]
                spectra = [config['data']['star_spectra'], config['data']['planet_spectra'][i]]
                obs = MedisObs(f'{i}', contrast, lods, spectra)
                photons = obs.photons

            c = NnReform(photons, outfile, train_type=train_type, aug_ind=aug_ind, debug=debugs[i], rm_input=obs.medis_cache)
            c.process_photons()
            c.save_class()

    workingdir_config = config['working_dir'] + 'config.yml'
    repo_config = os.path.join(os.path.dirname(__file__), 'config/config.yml')
    if not os.path.exists(workingdir_config):
        shutil.copyfile(repo_config, workingdir_config)

if __name__ == "__main__":
    make_input(config)
