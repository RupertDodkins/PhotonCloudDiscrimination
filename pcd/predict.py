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

# from examples.unet import UNet
from examples.minkunet import MinkUNet14A
from pcd.config.config import config
from pcd.input import load_dataset
from pcd.train import reform_input
from pcd.visualization import metric_tesseracts
from pcd.article_plots import pt_step


def predict():
    # net = UNet(in_nchannel=4, out_nchannel=2, D=4)
    net = MinkUNet14A(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    device = torch.device('cuda')
    net = net.to(device)
    print(f"loading NN from {config['savepath']}")
    net.load_state_dict(torch.load(config['savepath']))
    net.eval()

    # evalfiles = np.append(config['trainfiles'], config['testfiles'])
    evalfiles = config['testfiles']

    # for evalfile in config['mec']['glados_inputfiles']:
    for evalfile in evalfiles:
        coords, labels, astro_dict = load_dataset(evalfile)
        input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

        with torch.no_grad():
            output = net(input_pt)
        #     logits = output.slice(input_pt)
        # _, pred = logits.max(1)
        # pred = pred.cpu().numpy()
        pred = output.F.cpu().detach().numpy()

        pt_step(coords, np.int_(labels), pred, -1, astro_dict=astro_dict, train=False)

if __name__ == '__main__':
    predict()
    metric_tesseracts(start = 0, end = -1, jump=1, type='eval')