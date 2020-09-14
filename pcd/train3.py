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

from pcd.config.config import config

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

def train():

    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = MinkUNet14A(in_channels=4, out_channels=2, D=4)  # D is 4 - 1
    # net = MinkUNet34C(3, 2, 4)
    device = torch.device('cuda')
    net = net.to(device)
    # print(net)

    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        # from tests.common import data_loader
        # coords, feat, label = data_loader(is_classification=False)
        # print(coords.shape, feat.shape, label.shape)
        # print(coords, feat, label)

        coords, labels = load_dataset(config['trainfiles'][:1], config['train']['batch_size'])
        feats = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.]])[np.int_(labels)]
        # print(coords.shape, feats.shape, labels.shape)
        coords = np.concatenate((coords, np.zeros((140, coords.shape[1], 1))), axis=2)
        # print(coords.shape)
        # val_ds = load_dataset(config['testfiles'], config['train']['batch_size'])

        input = ME.SparseTensor(torch.from_numpy(feats[0]).float(), coords=torch.from_numpy(coords[0]*10000).float()).to(device)
        # print(input.type(), input.float().type())
        labels = torch.from_numpy(labels[0]).long().to(device)
        # print(type(labels), input.D, input.__repr__)
        # labels = labels

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, labels)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))

if __name__ == '__main__':
    train()

