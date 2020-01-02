import numpy as np

from medis.params import sp, ap, tp, iop, cp
import medis.save_photon_data as spd
from medis.Utils.misc import dprint
from medis.Utils.plot_tools import quicklook_im, view_datacube
import medis.Detector.readout as read

import medis_data

def reformat_planets(fields):
    """
    reformat the output of make_sixcube into what master.form() requires

    Takes a complex sixcube, removes the save_locs axis, modulos the complex data, sums the planets of object axis and
    stacks that four cube with the parent star fourcube to create a fivecube and overwrites iop.fields

    :param fields:
    :return:
    """
    obs_seq = np.abs(fields[:,-1]) ** 2
    dprint(fields.shape)
    tess = np.sum(obs_seq[:,:,1:], axis=2)
    # view_datacube(tess[0], logAmp=True, show=False)
    double_cube = np.zeros((ap.numframes, ap.w_bins, 2, ap.grid_size, ap.grid_size))
    double_cube[:, :, 0] = obs_seq[:, :, 0]
    collapse_comps = np.sum(obs_seq[:, :, 1:], axis=2)
    double_cube[:, :, 1] = collapse_comps
    # view_datacube(double_cube[0,:,0], logAmp=True, show=False)
    # view_datacube(double_cube[0,:,1], logAmp=True, show=True)
    print(f"Reduced shape of obs_seq = {np.shape(double_cube)} (numframes x nwsamp x 2 x grid x grid)")
    read.save_fields(double_cube, fields_file=iop.fields)
    return double_cube

def make_fields_master():
    """ The master fields file of which all the photons are seeded from according to their device_params

    :return:
    """

    medis_data.update_params()

    fields = spd.run_medis()
    tess = np.sum(fields, axis=2)
    view_datacube(tess[0], logAmp=True, show=False)
    view_datacube(tess[:,0], logAmp=True, show=True)

    # plt.plot(np.sum(fields, axis = (0,1,3,4)))
    # plt.show()
    dprint(fields.shape)
    if fields.shape[3] == len(ap.contrast)+1:
        fields = reformat_planets(fields)
    else:
        view_datacube(fields[0, :, 0], logAmp=True, show=False)
        view_datacube(fields[0, :, 1], logAmp=True, show=True)
        # view_datacube(fields[:, -1, 2], logAmp=True, show=False)


def make_data():
    make_fields_master()

if __name__ == "__main__":
    make_data()
