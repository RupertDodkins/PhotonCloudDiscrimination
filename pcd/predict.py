"""
Using a trained model, make predictions on unseen data

Inputs
Trained model
Photon set (Obsfile or simple photon table)

Settings
Use SSD
Field rotation
Wavefront modulation

Outputs
Photon Lables
"""
import numpy as np

import torch

from examples.unet import UNet
from pcd.config.config import config
from pcd.input import load_dataset
from pcd.train3 import reform_input
from pcd.visualization import tf_step


def predict():
    net = UNet(in_nchannel=4, out_nchannel=2, D=4)
    device = torch.device('cuda')
    net = net.to(device)
    net.load_state_dict(torch.load('test.pth'))
    net.eval()

    with torch.no_grad():
        for i in range(int(config['data']['num_indata'] * config['data']['test_frac'])):
            coords, labels = load_dataset(config['testfiles'][i:i + 1], config['train']['batch_size'])
            input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

            output = net(input_pt)

            tf_step(coords, np.int_(labels), output.F.cpu().detach().numpy(), train=False)

if __name__ == '__main__':
    predict()