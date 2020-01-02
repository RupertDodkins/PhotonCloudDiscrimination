import numpy as np
from medis.params import sp, ap, tp, iop, cp
import medis.save_photon_data as spd
import data

def make_fields_master():
    """ The master fields file of which all the photons are seeded from according to their device_params

    :return:
    """

    config.set_field_params()

    fields = spd.run_medis()
    # tess = np.sum(fields, axis=2)
    # view_datacube(tess[0], logAmp=True, show=False)
    # view_datacube(tess[:,0], logAmp=True, show=True)

    # plt.plot(np.sum(fields, axis = (0,1,3,4)))
    # plt.show()
    dprint(fields.shape)
    if fields.shape[3] == len(ap.contrast)+1:
        fields = reformat_planets(fields)
    # else:
    #     view_datacube(fields[0, :, 0], logAmp=True, show=False)
    #     view_datacube(fields[0, :, 1], logAmp=True, show=True)
    #     # view_datacube(fields[:, -1, 2], logAmp=True, show=False)


def make_data():
    make_fields_master()

if __name__ == "__main__":
    make_data()