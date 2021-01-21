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
import random

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
            self.filenames = glob.glob(config['mec']['dark_data']+'*.h5')

        self.filenames = self.filenames[::5]

        # if debug: fig = plt.figure(figsize=(12,12))

        for i, filename in enumerate(self.filenames):
            iop.photonlist = filename

            print(i, filename)

            cam = Camera(usesave=False, product='photons')
            cam.load_photontable()  # don't give it option to create photons

            photons = cam.photons

            print('photons', len(photons[0]))

            tcut = np.logical_and(photons[0] > 0, photons[0] < 15)
            photons = photons[:, tcut]
            if not config['mec']['dithered']:
                photons[0] += int(self.filenames[i][:-3].split('/')[-1]) - int(self.filenames[0][:-3].split('/')[-1])

            print('photons', len(photons[0]))

            photons[1] = cam.phase_cal(photons[1]/1e9)  #angstroms
            pcut = np.logical_and(photons[1] > -150, photons[1] < 0)
            photons = photons[:, pcut]

            # pcut = np.logical_and(photons[1] < 1e5, photons[1] > 100)
            # pcut = np.logical_and(photons[1] > 0, photons[1] < 1e4)
            # photons = photons[:, pcut]

            # print('photons', len(photons[0]))

            # if debug:
            #     ax = fig.add_subplot(6,6,i+1)
            #     H, xb, yb = np.histogram2d(photons[2], photons[3], bins=100)
            #     ax.imshow(H)

            print('photons', len(photons[0]))

            # chosen = sample(range(len(photons[0])),
            #                 config['data']['num_indata'] * int(np.ceil(1.5 * config['num_point'] / len(self.filenames))))
            # photons = photons[:, chosen]

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

    outfiles = np.append(config['trainfiles'], config['testfiles'])
    # outfiles = [config['mec']['dark_data'] + file.split('/')[-1] for file in outfiles]

    debugs = [False] * config['data']['num_indata']
    # debugs[0] = True
    train_types = ['train'] * config['data']['num_indata']
    num_test = config['data']['num_indata'] * config['data']['test_frac']
    num_test = int(num_test)
    if num_test > 0: train_types[-num_test:] = ['test']*num_test

    obs = MecObs(config['mec']['h5s'])
    photons = obs.photons

    astro_params = input.MedisParams(config)

    if config['mec']['dithered']:
        mec_numframes = 1
        # mec_sample_time = int(obs.filenames[1][:-9].split('/')[-1]) - int(obs.filenames[0][:-9].split('/')[-1])
        mec_sample_time = 6
    else:
        mec_numframes = len(obs.filenames)
        mec_sample_time = int(obs.filenames[1][:-3].split('/')[-1]) - int(obs.filenames[0][:-3].split('/')[-1])

    medis_numframes = copy.copy(sp.numframes)
    medis_sample_time = copy.copy(sp.sample_time)

    for i, outfile, train_type in zip(range(config['data']['num_indata']), outfiles, train_types):
        if not os.path.exists(outfile):

            if inject_fake_comp:
                astro = astro_params(i)
                med_ind = np.arange(config['data']['num_indata'])[i] // (config['data']['aug_ratio'] + 1)
                sp.numframes = medis_numframes
                sp.sample_time = medis_sample_time
                obs = input.MedisObs(f'{med_ind}', astro, debug=False)
                obs.adjust_companion()
                planet_photons = obs.photons[1]
                filephotons = [photons[0], planet_photons]

            elif config['mec']['companion_coords']:
                print('moving companion')
                astro = astro_params(i)
                contrast = [astro[0]]
                companion_xy = [astro[1]]  #todo implement using this
                spectra = astro[2]
                xc,yc,rc = config['mec']['companion_coords']
                filephotons = copy.copy(photons[0])

                # get total number of photons at planet
                tot_aper_inds = (filephotons[:,2] - yc) ** 2 + (filephotons[:,3] - xc) ** 2  <= rc ** 2
                num_tot = len(np.where(tot_aper_inds)[0])

                # get number of star photons
                num_star = 0
                for a in [-1,1]:
                    for b in [-1,1]:
                        num_star += len(np.where(((filephotons[:,2]-xc-a*2*rc)**2) + ((filephotons[:,3]-yc-b*2*rc)**2) <= (rc**2))[0])
                num_star /= 4

                # select indicies for planet photons
                num_planet = int((num_tot - num_star)*1.5)  # fudge factor to make sure at least most of planet photons are identified
                phase = filephotons[tot_aper_inds,1]
                wsamples = (phase - mp.wavecal_coeffs[1]) / (mp.wavecal_coeffs[0])

                # assign probability based on spectrum
                spectrum = planck(spectra[1], wsamples)
                spec_pdf = spectrum/np.sum(spectrum)

                # assign probability based on location
                dists = np.sqrt((filephotons[tot_aper_inds, 2] - yc) ** 2 + (filephotons[tot_aper_inds, 3] - xc) ** 2)
                dist_pdf = 1./dists
                dist_pdf /= np.sum(dist_pdf)

                # create correction for unequal original spectral distribution (already did correction for spatial)
                bins = np.linspace(phase.min(), phase.max(), 100)
                H, _ = np.histogram(phase, bins)
                idx = np.digitize(phase, bins)  # H[idx] gives the bin heights assigned to each photon
                tot_phase = H[idx-2]
                tot_phase = tot_phase/np.sum(tot_phase)

                # combine the assigned probabilities
                pdf = (spec_pdf + dist_pdf) / 2
                pdf /= tot_phase  # larger original bin heights require smaller weighting to cancel out sampling effect
                pdf /= np.sum(pdf)

                planet_inds = np.random.choice(np.where(tot_aper_inds)[0], size=num_planet, p=pdf)

                # separate star and planet photons
                planet_photons = filephotons[planet_inds]
                star_photons = np.delete(filephotons, planet_inds, axis=0)

                ratio = contrast[0] / 10. ** config['data']['contrasts'][0]
                assert ratio <= 1
                if ratio < 1:
                    del_planet_inds = np.sort(random.sample(range(len(planet_inds)), int(len(planet_inds) * (1 - ratio))))
                    planet_photons = np.delete(planet_photons, del_planet_inds, axis=0)

                print(contrast, len(planet_inds), len(planet_photons))
                # rotate planet photons around star
                offset = np.random.uniform(-90,15,1)[0]
                rad_offset = np.random.uniform(-10,10,1)[0]

                ystar, xstar = np.array(mp.array_size)//2
                centered_y, centered_x = planet_photons[:,2] - ystar, planet_photons[:,3] - xstar
                centered_y += rad_offset if yc > 0 else -rad_offset
                centered_x += rad_offset if xc > 0 else -rad_offset

                num_planet = len(planet_photons)
                for p in range(num_planet):
                    angle = np.deg2rad(planet_photons[p,0] * config['data']['rot_rate'] + offset)
                    rot_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                    centered_y[p], centered_x[p] = np.dot(rot_matrix, np.array([centered_y[p], centered_x[p]]).T)
                    planet_photons[p, 2], planet_photons[p, 3] = centered_y[p] + ystar, centered_x[p] + xstar
                filephotons = [star_photons, planet_photons]

            else:
                astro = None
                filephotons = [photons[0], np.array([photons[0][0]])]

            filephotons = [filephotons[o][i::config['data']['num_indata']] for o in range(2)]

            sp.numframes = mec_numframes
            sp.sample_time = mec_sample_time
            c = input.NnReform(filephotons, outfile, train_type=train_type, debug=debugs[i],
                               dithered=config['mec']['dithered'], astro=astro)
            c.process_photons()
            c.save_class()

    workingdir_config = config['mec']['dark_data'] + 'config.yml'
    repo_config = os.path.join(os.path.dirname(__file__), 'config/config.yml')
    if not os.path.exists(workingdir_config):
        shutil.copyfile(repo_config, workingdir_config)

if __name__ == '__main__':
    make_input(config)


