""" Code to train ME """

import os
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C, MinkUNet14A
# from examples.unet import UNet

from pcd.config.config import config
from pcd.visualization import tf_step, pt_step
from pcd.input import load_dataset

if os.path.exists(config['train']['pt_outputs']):
    os.remove(config['train']['pt_outputs'])

def reform_input(coords, labels, device):
    labels = np.int32(labels)
    feats = coords

    coords, feats, labels = coords[0], feats[0], labels[0]
    if config['data']['quantize']:
        quantization_size = 0.005 * 10**((config['num_point'] / 131072) - 1)  # = 0.005 for num_point==131072, 0.05 for num_point== 262144
        coords, feats, labels = ME.utils.sparse_quantize(
            coords=coords,
            feats=feats,
            labels=labels,
            quantization_size=quantization_size)

    if config['data']['batch_coords']:
        coords_pt = ME.utils.batched_coordinates([coords * 1e6])
    else:
        coords = np.concatenate((coords * 1e6, np.zeros((coords.shape[0], 1))), axis=1)
        coords_pt = torch.from_numpy(coords).int()
    feats_pt =  torch.from_numpy(feats).float()

    labels_pt = torch.from_numpy(labels).long().to(device)

    input_pt = ME.SparseTensor(feats_pt,
                               coords=coords_pt,
                               quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
                               ).to(device)

    return input_pt, labels_pt, coords, feats, labels

def train():

    # net = UNet(in_nchannel=4, out_nchannel=2, D=4)  # D is 4 - 1
    net = MinkUNet14A(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    device = torch.device('cuda')
    net = net.to(device)

    # loss and network
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1], device=device))

    if os.path.exists(config['savepath']):
        print('Loading neural net')
        net.load_state_dict(torch.load(config['savepath']))

    optimizer = SGD(net.parameters(), lr=1e-4)
    for epoch in range(config['train']['max_epoch']):
        print('epoch: ', epoch)
        net.train()
        for i in range(int(config['data']['num_indata']*(1-config['data']['test_frac']))):
            optimizer.zero_grad()

            # Get new data
            coords, labels = load_dataset(config['trainfiles'][i:i+1])
            input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

            # Forward
            output = net(input_pt)

            # Loss
            loss = criterion(output.F, labels_pt)

            # if i % 5 == 0:
            pt_step(coords, np.int_(labels), output.F.cpu().detach().numpy(), loss.item(), train=True)

            # Gradient
            loss.backward()
            optimizer.step()

        test(net, device, criterion)

        # Saving and loading a network
        torch.save(net.state_dict(), config['savepath'])

def test(net, device, criterion):
    net.eval()
    with torch.no_grad():
        for i in range(int(config['data']['num_indata'] * config['data']['test_frac'])):
            coords, labels = load_dataset(config['testfiles'][i:i + 1])
            input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

            output = net(input_pt)

            loss = criterion(output.F, labels_pt)

            # if i % 10 == 0:
            pt_step(coords, np.int_(labels), output.F.cpu().detach().numpy(), loss.item(), train=False)

if __name__ == '__main__':
    train()


