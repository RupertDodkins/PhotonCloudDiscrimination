""" Code to train ME """

import os
import numpy as np
import matplotlib.pyplot as plt

import h5py
import open3d as o3d

import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C, MinkUNet14A
from examples.resnet import ResNet50
from examples.unet import UNet

from pcd.config.config import config
from pcd.evaluate import tf_step

if os.path.exists(config['train']['outputs']):
    os.remove(config['train']['outputs'])

def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    smpw = f['smpw'][:]
    return (data, label, smpw)

def load_dataset(in_files, batch_size):
    for in_file in in_files:
        assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 1000

    in_data = np.empty((0, config['num_point'], config['dimensions']))
    in_label = np.empty((0, config['num_point']))
    class_weights = []
    for in_file in in_files:
        print(f'loading {in_file}')
        file_in_data, file_in_label, class_weights = load_h5(in_file)
        in_data = np.concatenate((in_data, file_in_data), axis=0)
        in_label = np.concatenate((in_label, file_in_label), axis=0)

    return in_data, in_label

def reform_input(coords, labels, device):
    labels = np.int32(labels)
    feats = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.]])[labels]

    # coords = np.concatenate((coords, np.zeros((coords.shape[0], coords.shape[1], 1))), axis=2)
    #print(warning the batch num should be on left)
    # input_pt = ME.SparseTensor(torch.from_numpy(feats[0]).float(),
    #                            coords=torch.from_numpy(coords[0] * 1e6).float()).to(device)

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
                               coords=coords_pt
                               ).to(device)

    return input_pt, labels_pt, coords, feats, labels

def train():

    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = UNet(in_nchannel=4, out_nchannel=2, D=4)  # D is 4 - 1
    # net = ResNet50(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    # net = MinkUNet14A(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    device = torch.device('cuda')
    net = net.to(device)
    # print(net)

    optimizer = SGD(net.parameters(), lr=1e-2)
    for epoch in range(config['train']['max_epoch']):

        for i in range(int(config['data']['num_indata']*(1-config['data']['test_frac']))):
            optimizer.zero_grad()

            # Get new data
            coords, labels = load_dataset(config['trainfiles'][i:i+1], config['train']['batch_size'])
            input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

            # Forward
            output = net(input_pt)

            # Loss
            loss = criterion(output.F, labels_pt)
            print('Iteration: ', i, ', Loss: ', loss.item())

            tf_step(coords, np.int_(labels), output.F.cpu().detach().numpy(), train=True)

            # Gradient
            loss.backward()
            optimizer.step()

        test(net, device)

        # Saving and loading a network
        torch.save(net.state_dict(), 'test.pth')
    # net.load_state_dict(torch.load('test.pth'))

def test(net, device):
    for i in range(int(config['data']['num_indata'] * config['data']['test_frac'])):
        coords, labels = load_dataset(config['testfiles'][i:i + 1], config['train']['batch_size'])
        input_pt, labels_pt, coords, _, labels = reform_input(coords, labels, device)

        output = net(input_pt)

        # loss = criterion(output.F, labels_pt)
        # print('Iteration: ', i, ', Loss: ', loss.item())

        tf_step(coords, np.int_(labels), output.F.cpu().detach().numpy(), train=False)

    # net = MinkUNet14A(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    # net.load_state_dict(torch.load('test.pth'))

if __name__ == '__main__':
    train()
    # test()

