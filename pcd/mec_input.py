import os
import shutil
import numpy as np
from random import sample

from medis.MKIDS import Camera

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config
import pcd.input as input


class MecObs():
    """ Gets the photon lists from photontables """
    def __init__(self, filenames, debug=True):
        mp.array_size = [144, 140]  # the size of mec arrays
        # ap.wvl_range = np.array([0,1500])/1e9

        iop.update_testname(config['working_dir'])
        self.display_2d_hists = input.MedisObs.display_2d_hists
        self.numobj = 1
        self.medis_cache = iop.testdir

        self.photons = np.empty((0, 4))

        for filename in filenames:
            iop.photonlist = os.path.join(config['working_dir'], filename)

            cam = Camera(usesave=False, product='photons')
            cam.load_photontable()  # don't give it option to create photons

            photons = cam.photons

            print('photons', len(photons[0]))

            tcut = np.logical_and(photons[0] > 0, photons[0] < 11)
            photons = photons[:, tcut]

            print('photons', len(photons[0]))

            photons[1] = cam.phase_cal(photons[1]/1e10)  #angstroms
            pcut = np.logical_and(photons[1] > -250, photons[1] < 0)
            photons = photons[:, pcut]

            # pcut = np.logical_and(photons[1] < 1e5, photons[1] > 100)
            # pcut = np.logical_and(photons[1] > 0, photons[1] < 1e4)
            # photons = photons[:, pcut]

            print('photons', len(photons[0]))

            photons[2] += config['mec']['offset'][0]
            photons[3] += config['mec']['offset'][1]

            xcut = np.logical_and(photons[2] > 0, photons[2] < mp.array_size[1])
            photons = photons[:, xcut]

            print('photons', len(photons[0]))

            ycut = np.logical_and(photons[3] > 0, photons[3] < mp.array_size[0])
            photons = photons[:, ycut]

            print('photons', len(photons[0]))

            chosen = sample(range(len(photons[0])), int(np.ceil(config['num_point']/len(filenames))))
            photons = photons[:, chosen]

            print('photons', len(photons[0]))

            photons = photons.T

            self.photons = np.concatenate((self.photons, photons), axis=0)

        self.photons = [self.photons]  # give it an extra dim to be plot by disp2dhist
        if debug:
            self.display_2d_hists(self)


def make_input(config, inject_fake_comp=False):
    if inject_fake_comp:
        d = input.Data(config)

    # outfiles = np.append(config['trainfiles'], config['testfiles'])
    outfiles = config['mec']['evalfiles']

    debugs = [False] * config['data']['num_indata']
    debugs[0] = True
    train_types = ['train'] * config['data']['num_indata']
    num_test = config['data']['num_indata'] * config['data']['test_frac']
    num_test = int(num_test)
    if num_test > 0: train_types[-num_test:] = ['test']*num_test

    for i, outfile, train_type in zip(range(config['data']['num_indata']), outfiles, train_types):
        if not os.path.exists(outfile):
            obs = MecObs(config['mec']['h5s'])
            photons = obs.photons

            if inject_fake_comp:
                contrast = [d.contrasts[i]]
                lods = [d.lods[i]]
                spectra = [config['data']['star_spectra'], config['data']['planet_spectra'][i]]
                obs = input.MedisObs(f'{i}', contrast, lods, spectra)
                planet_photons = obs.photons[1]
                photons = [photons[0], planet_photons]

            c = input.NnReform(photons, outfile, train_type=train_type, debug=debugs[i],
                           rm_input=obs.medis_cache)
            c.process_photons()
            c.save_class()

    workingdir_config = config['working_dir'] + 'config.yml'
    repo_config = os.path.join(os.path.dirname(__file__), 'config/config.yml')
    if not os.path.exists(workingdir_config):
        shutil.copyfile(repo_config, workingdir_config)

if __name__ == '__main__':
    make_input(config)


