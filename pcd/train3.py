# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os

import argparse
import numpy as np
from urllib.request import urlretrieve
import h5py
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.common import Timer
from pcd.config.config import config

# # Check if the weights and file exist and download
# if not os.path.isfile('weights.pth'):
#     print('Downloading weights and a room ply file...')
#     urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/weights.pth",
#                 'weights.pth')
#     urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", '1.ply')
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--file_name', type=str, default='1.ply')
# parser.add_argument('--weights', type=str, default='weights.pth')
# parser.add_argument('--use_cpu', action='store_true')
#
# CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
#                 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
#                 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
#                 'bathtub', 'otherfurniture')
#
# VALID_CLASS_IDS = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
# ]
#
# SCANNET_COLOR_MAP = {
#     0: (0., 0., 0.),
#     1: (174., 199., 232.),
#     2: (152., 223., 138.),
#     3: (31., 119., 180.),
#     4: (255., 187., 120.),
#     5: (188., 189., 34.),
#     6: (140., 86., 75.),
#     7: (255., 152., 150.),
#     8: (214., 39., 40.),
#     9: (197., 176., 213.),
#     10: (148., 103., 189.),
#     11: (196., 156., 148.),
#     12: (23., 190., 207.),
#     14: (247., 182., 210.),
#     15: (66., 188., 102.),
#     16: (219., 219., 141.),
#     17: (140., 57., 197.),
#     18: (202., 185., 52.),
#     19: (51., 176., 203.),
#     20: (200., 54., 131.),
#     21: (92., 193., 61.),
#     22: (78., 71., 183.),
#     23: (172., 114., 82.),
#     24: (255., 127., 14.),
#     25: (91., 163., 138.),
#     26: (153., 98., 156.),
#     27: (140., 153., 101.),
#     28: (158., 218., 229.),
#     29: (100., 125., 154.),
#     30: (178., 127., 135.),
#     32: (146., 111., 194.),
#     33: (44., 160., 44.),
#     34: (112., 128., 144.),
#     35: (96., 207., 209.),
#     36: (227., 119., 194.),
#     37: (213., 92., 176.),
#     38: (94., 106., 211.),
#     39: (82., 84., 163.),
#     40: (100., 85., 144.),
# }
#
# import matplotlib.pylab as plt
#
#
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
    # config = parser.parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f"Using {device}")
    # Define a model and load the weights
    model = MinkUNet34C(3, 2, 4).to(device)
    # model_dict = torch.load('weights.pth')
    # model.load_state_dict(model_dict)
    model.eval()

    # coords, colors, pcd = load_file(config.file_name)
    # coords, labels, pcd = load_file('/home/dodkins/PythonProjects/Open3D/examples/test_data/fragment.ply')
    # print(coords.shape, labels.shape, 'shapes')
    coords, labels = load_dataset(config['trainfiles'][:1], config['train']['batch_size'])
    colors = np.array([[0.,0.,0.], [1.,1.,1.]])[np.int_(labels)]
    # val_ds = load_dataset(config['testfiles'], config['train']['batch_size'])
    print(coords.shape, colors.shape)
    # Measure time
    with torch.no_grad():
        voxel_size = 0.02

        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            feats=torch.from_numpy(colors[0]).float(),
            coords=ME.utils.batched_coordinates([coords[0] / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        ).to(device)
        logits = model(sinput).slice(sinput)

    _, pred = logits.max(1)
    pred = pred.cpu().numpy()
    print(pred.shape, np.max(pred), np.min(pred))
    plt.plot(labels[0], pred)
    plt.show()

    # # # Create a point cloud file
    # # pred_pcd = o3d.geometry.PointCloud()
    # # # Map color
    # # colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
    # # pred_pcd.points = o3d.utility.Vector3dVector(coords)
    # # pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    #
    # # Move the original point cloud
    # # pcd.points = o3d.utility.Vector3dVector(
    # #     np.array(pcd.points) + np.array([0, 5, 0]))
    #
    # # print(np.asarray(pcd.points).shape, np.asarray(pred_pcd.points).shape)
    #
    # # o3d.visualization.draw_geometries([pcd, pred_pcd])
    #
    # # Visualize the input point cloud and the prediction
    # # vis = o3d.visualization.Visualizer()
    # # vis.create_window()
    # # print('here')
    # # print(np.asarray(pcd.points).shape)
    # # vis.add_geometry(pcd)
    # # print(np.asarray(pcd.points).shape)
    # #
    # # vis.capture_screen_image("temp_%04d.jpg" % 0)
    # # depth = vis.capture_depth_float_buffer(False)
    # # image = vis.capture_screen_float_buffer(False)
    # #
    # #
    # # plt.imshow(depth)
    # # plt.figure()
    # # plt.imshow(image)
    # # plt.show()
    #
    # vis.run()
    # print('ran')
    # vis.destroy_window()
    # print('window distroyed')
    # # o3d.visualization.draw_geometries([pcd, pred_pcd])
    #

if __name__ == '__main__':
    train()

