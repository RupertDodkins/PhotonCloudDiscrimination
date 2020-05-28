This package allows for training and evaluating nets that perform the PCD algorithm.

Photon Cloud Discrimination (PCD) is an algorithm that given a set of photon coordinates in xytw space, applies a label to each photon for the source type (object segementation) and identifies the most likely location of companion sources in the focal plane.

## Installation

##### Modules

conda env create -f medis_env.yml  #todo change tensorflow==1.13.1 and see if still works rather than postfacto pip upgrade

conda activate medis-tf

cd proper_v3.2.1_python_3.x

python setup.py install

##### Tensorflow ops

Check the paths are correct in the shell scripts

To see images if no DISPLAY variable set after ssh -X, try gedit command and close the window, then echo $DISPLAY


