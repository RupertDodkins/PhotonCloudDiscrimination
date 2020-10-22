This package allows for training and evaluating nets that perform the PCD algorithm.

Photon Cloud Discrimination (PCD) is an algorithm that given a set of photon coordinates in xytw space, applies a label to each photon for the source type (object segementation) and identifies the most likely location of companion sources in the focal plane.

## Installation

##### Modules

conda env create -f environment.yml

conda activate medis-tf

export CUDA_HOME="/usr/local/cuda-10.2"

conda install numpy mkl-include pytorch cudatoolkit=10.2 -c pytorch

git clone https://github.com/NVIDIA/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install

cd ../proper_v3.2.1_python_3.x_12feb20

python setup.py install


##### Debug

To see images if no DISPLAY variable set after ssh -X, try gedit command and close the window, then echo $DISPLAY
