"""
MEDIS configuration file

"""

import os
import numpy as np
from medis.params import sp, ap, tp, iop, mp, atmp
from pcd.config.config import config

iop.datadir = config['working_dir']
# iop.photonlist = os.path.join(iop.datadir, 'photonlist.pkl')
# print(iop.photonlist, 'photonlist')

sp.sample_time = 30  # 0.5
sp.numframes = 20
sp.grid_size = 512
sp.num_processes = 7
sp.beam_ratio = 0.15
sp.debug = False
sp.quick_detect = False
sp.save_sim_object = False
sp.save_to_disk = True
sp.checkpointing = 10
sp.num_processes = 3

ap.companion = True

if config['model'] != 'minkowski':  # minkowski only has one pointcloud per input data so requires less photons
    ap.star_flux = 1e7/1.8 * config['num_point']/65536.
else:
    ap.star_flux = 1e5/1.8 * config['num_point']/65536.  # 1e5 flux is sufficient for a 65536 point point-cloud
ap.spectra = None
ap.contrast = [10**-2]
ap.companion_xy = [[2.5,0]]
ap.n_wvl_init = 4
ap.n_wvl_final = 8
ap.wvl_range = np.array([800, 1500]) / 1e9

atmp.model = 'single'

tp.obscure = False
tp.satelite_speck['apply'] = True
tp.satelite_speck['amp'] = 12e-10
tp.entrance_d = 8.
tp.use_ao = True
tp.ao_act = 50
tp.use_atmos = True
tp.prescription = 'general_telescope'
tp.cg_type = 'Solid'
tp.rot_rate = 0  #0  # do rotation with input.py now   1./60  # 9  # deg/s
tp.pix_shift = [0, 0]

mp.array_size = np.array([150,150])
mp.wavecal_coeffs = [1.e9 / 6, -250]
mp.hot_counts = False
mp.dark_counts = False
mp.platescale = 10 * 1e-3
