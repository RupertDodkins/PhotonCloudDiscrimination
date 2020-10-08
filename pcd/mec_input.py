""" This script creates input data from mec h5s. Run this by switching python interpreter to dark pipeline-medis
environment, set default deployment to dark, deploy and set dir in config.yml. Creates the input files and then scp
 to glados """

import os
import shutil
import numpy as np
from random import sample
import glob
import matplotlib.pyplot as plt
import copy

from medis.MKIDS import Camera
from medis.distribution import planck

from pcd.config.medis_params import sp, ap, tp, iop, mp
from pcd.config.config import config
import pcd.input as input


class MecObs():
    """ Gets the photon lists from photontables """
    def __init__(self, filenames, debug=False):
        mp.array_size = [144, 140]  # the size of mec arrays
        # ap.wvl_range = np.array([0,1500])/1e9

        iop.update_testname(config['mec']['dark_data'])
        self.display_2d_hists = input.MedisObs.display_2d_hists
        self.numobj = 1
        self.medis_cache = iop.testdir

        self.photons = np.empty((0, 4))

        if filenames is None:
            filenames = glob.glob(config['mec']['dark_data']+'*.h5')[::6]

        if debug: fig = plt.figure(figsize=(12,12))

        for i, filename in enumerate(filenames):
            iop.photonlist = filename

            print(i, filename)

            cam = Camera(usesave=False, product='photons')
            cam.load_photontable()  # don't give it option to create photons

            photons = cam.photons

            print('photons', len(photons[0]))

            tcut = np.logical_and(photons[0] > 0, photons[0] < 11)
            photons = photons[:, tcut]

            print('photons', len(photons[0]))

            photons[1] = cam.phase_cal(photons[1]/1e9)  #angstroms
            pcut = np.logical_and(photons[1] > -150, photons[1] < 0)
            photons = photons[:, pcut]

            # pcut = np.logical_and(photons[1] < 1e5, photons[1] > 100)
            # pcut = np.logical_and(photons[1] > 0, photons[1] < 1e4)
            # photons = photons[:, pcut]

            # print('photons', len(photons[0]))

            if debug:
                ax = fig.add_subplot(6,6,i+1)
                H, xb, yb = np.histogram2d(photons[2], photons[3], bins=150)
                ax.imshow(H)

            print('photons', len(photons[0]))

            chosen = sample(range(len(photons[0])),
                            config['data']['num_indata'] * int(np.ceil(1.5 * config['num_point'] / len(filenames))))
            photons = photons[:, chosen]

            print('photons', len(photons[0]))

            photons = photons.T
            self.photons = np.concatenate((self.photons, photons), axis=0)

        # convert to pix values
        if config['mec']['dithered']:
            self.photons[:,2] = (self.photons[:,2] - self.photons[:,2].min()) * 3600 / mp.platescale
            self.photons[:,3] = (self.photons[:,3] - self.photons[:,3].min()) * 3600 / mp.platescale

        boarders = self.photons[:,2].max(), self.photons[:,3].max()
        #
        self.photons[:,2] += config['mec']['offset'][0]
        self.photons[:,3] += config['mec']['offset'][1]

        print(boarders)
        #
        xcut = np.logical_and(self.photons[:,2] > 0, self.photons[:,2] < boarders[0])
        self.photons = self.photons[xcut]

        print('self.photons', len(self.photons[:,0]))

        ycut = np.logical_and(self.photons[:,3] > 0, self.photons[:,3] < boarders[1])
        self.photons = self.photons[ycut]

        print('self.photons', len(self.photons[:,0]))

        self.photons = [self.photons]  # give it an extra dim to be plot by disp2dhist

        if debug:
            plt.tight_layout()
            plt.show(block=False)
            self.display_2d_hists(self)

def make_input(config, inject_fake_comp=False):
    if inject_fake_comp:
        d = input.Data(config)

    outfiles = np.append(config['trainfiles'], config['testfiles'])
    outfiles = [config['mec']['dark_data'] + file.split('/')[-1] for file in outfiles]

    debugs = [False] * config['data']['num_indata']
    debugs[0] = True
    train_types = ['train'] * config['data']['num_indata']
    num_test = config['data']['num_indata'] * config['data']['test_frac']
    num_test = int(num_test)
    if num_test > 0: train_types[-num_test:] = ['test']*num_test

    obs = MecObs(config['mec']['h5s'])
    photons = obs.photons
    for i, outfile, train_type in zip(range(config['data']['num_indata']), outfiles, train_types):
        if not os.path.exists(outfile):

            if inject_fake_comp:
                contrast = [d.contrasts[i]]
                lods = [d.lods[i]]
                spectra = [config['data']['star_spectra'], config['data']['planet_spectra'][i]]
                obs = input.MedisObs(f'{i}', contrast, lods, spectra)
                planet_photons = obs.photons[1]
                filephotons = [photons[0], planet_photons]

            if config['mec']['companion_coords']:
                xc,yc,rc = config['mec']['companion_coords']
                print('moving companion')
                filephotons = copy.copy(photons[0])

                # get total number of photons at planet
                tot_aper_inds = (filephotons[:,2] - yc) ** 2 + (filephotons[:,3] - xc) ** 2  <= rc ** 2
                num_tot = len(np.where(tot_aper_inds)[0])

                # get number of star photons
                ystar, xstar = (filephotons[:,2].min() + filephotons[:,2].max())/2 , (filephotons[:,3].min() + filephotons[:,3].max())/2
                yconj = ystar - (yc-ystar) if yc - ystar > 0 else (ystar - yc) + ystar  # find opposite spot
                xconj = xc - 2*rc
                star_conj_inds = ((filephotons[:,2] - yconj) ** 2) + ((filephotons[:,3] - xconj) ** 2)  <= (rc ** 2)
                num_star = len(np.where(star_conj_inds)[0])

                # select indicies for planet photons
                num_planet = int(num_tot - num_star)
                # wsamples = np.linspace(ap.wvl_range[0], ap.wvl_range[1], 50)
                phase = filephotons[tot_aper_inds,1]
                wsamples = (phase - mp.wavecal_coeffs[1]) / (mp.wavecal_coeffs[0])
                spectrum = planck(config['data']['planet_spectra'][0], wsamples)
                pdf = spectrum/max(spectrum)
                # normed_phase = filephotons[tot_aper_inds,1] / filephotons[tot_aper_inds,1].min()
                # samples = np.random.choice(phase, size=num_planet, p=pdf)
                seeds = np.random.uniform(0,1,len(pdf))
                chosen = seeds<pdf
                # chosen = sample(range(num_tot), num_planet)
                planet_inds = np.where(tot_aper_inds)[0][chosen]

                # separate star and planet photons
                planet_photons = filephotons[planet_inds]
                num_planet = len(planet_photons)
                star_photons = np.delete(filephotons, planet_inds, axis=0)

                # rotate planet photons around star
                centered_y, centered_x = planet_photons[:,2] - ystar, planet_photons[:,3] - xstar
                offset = np.random.uniform(0,360,1)[0]
                for p in range(num_planet):
                    angle = np.deg2rad(planet_photons[p,0] * tp.rot_rate + offset)
                    rot_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]

                    centered_y[p], centered_x[p] = np.dot(rot_matrix, np.array([centered_y[p], centered_x[p]]).T)
                    planet_photons[p, 2], planet_photons[p, 3] = centered_y[p] + ystar, centered_x[p] + xstar
                filephotons = [star_photons, planet_photons]

            filephotons = [filephotons[o][i::config['data']['num_indata']] for o in range(2)]

            print(debugs, 'debug')
            c = input.NnReform(filephotons, outfile, train_type=train_type, debug=debugs[i],
                           rm_input=obs.medis_cache, dithered=config['mec']['dithered'])
            c.process_photons()
            c.save_class()

    workingdir_config = config['mec']['dark_data'] + 'config.yml'
    repo_config = os.path.join(os.path.dirname(__file__), 'config/config.yml')
    if not os.path.exists(workingdir_config):
        shutil.copyfile(repo_config, workingdir_config)

if __name__ == '__main__':
    make_input(config)


