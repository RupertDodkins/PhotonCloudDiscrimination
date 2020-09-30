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
from pcd.visualization import pt_step


def predict():
    net = UNet(in_nchannel=4, out_nchannel=2, D=4)
    device = torch.device('cuda')
    net = net.to(device)
    net.load_state_dict(torch.load(config['savepath']))
    net.eval()

    for evalfile in config['mec']['evalfiles']:
        coords, labels = load_dataset([evalfile])
        input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

        with torch.no_grad():
            output = net(input_pt)
        #     logits = output.slice(input_pt)
        # _, pred = logits.max(1)
        # pred = pred.cpu().numpy()
        pred = output.F.cpu().detach().numpy()

        pt_step(coords, np.int_(labels), pred, -1, train=False)

if __name__ == '__main__':
    predict()