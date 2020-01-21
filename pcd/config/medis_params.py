"""
MEDIS configuration file

Todo this should be a yml?
"""

import numpy as np
from medis.params import sp, ap, tp, iop, mp
from config.config import config

# def update_params():
iop.datadir = config['working_dir']
iop.set_atmosdata('200102')
iop.set_aberdata('Subaru')
iop.set_testdir('')

ap.sample_time = 0.05
ap.numframes = 20
sp.uniform_flux = False
sp.show_wframe = False
sp.save_obs = True
sp.show_cube = False
sp.num_processes = 1
sp.save_fields = False
sp.save_ints = True
sp.cont_save = False
# sp.save_locs = ['deformable_mirror', 'detector']

ap.companion = True
ap.star_photons_per_s = int(1e3)
ap.grid_size = 512
tp.beam_ratio = 0.25
# ap.contrast = 10**np.array([-3.5, -4, -4.5, -5] * 2)
# ap.lods = [[2.5,0], [0,3], [-3.5,0], [0,-4], [4.5,0], [0,5], [-5.5,0],[0,-6]]
# ap.nwsamp = 8
# ap.w_bins = 16
ap.contrast = [10**-2]
ap.lods = [[2.5,0]]
ap.nwsamp = 3
ap.w_bins = 8

# sp.save_locs = np.empty((0, 1))
tp.obscure = False
tp.satelite_speck = False
tp.diam = 8.
tp.use_ao = True
tp.include_tiptilt = False
tp.ao_act = 50
tp.platescale = 10  # mas
tp.detector = 'MKIDs'  #'ideal'
tp.use_atmos = True
tp.use_zern_ab = False
tp.occulter_type = 'Gaussian'
tp.aber_params = {'CPA': True,
                  'NCPA': True,
                  'QuasiStatic': False,  # or Static
                  'Phase': True,
                  'Amp': False,
                  'n_surfs': 4,
                  'OOPP': False}  # [16,8,4,16]}#False}#
tp.aber_vals = {'a': [5e-15, 1e-16],  # 'a': [5e-17, 1e-18],
                'b': [0.2, 0.02],
                'c': [3.1, 0.5],
                'a_amp': [0.05, 0.01]}
tp.piston_error = False
ap.band = np.array([800, 1500])
tp.rot_rate = 0  # deg/s
tp.pix_shift = [[0, 0]]

mp.array_size = np.array([150,150])
mp.wavecal_coeffs = [1./6, -250]
