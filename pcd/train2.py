""" Code adapted from pointnet2-tensorflow2.train_scannet """

import os
import sys
import numpy as np

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.sem_seg_model import SEM_SEG_Model
# from model_scannet import PointConvModel

tf.random.set_seed(42)

import h5py
from pcd.config.config import config

config['log_freq'] = 10
config['test_freq'] = 100
config['num_classes'] = config['classes']
config['lr'] = config['train']['learning_rate']
config['bn'] = False

if os.path.exists(config['train']['outputs']):
    os.remove(config['train']['outputs'])

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

    dataset = tf.data.Dataset.from_tensor_slices((in_data, in_label, np.array(class_weights)[np.int_(in_label)]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def train():
    model = SEM_SEG_Model(config['train']['batch_size'], config['num_classes'], config['bn'])
    # model = PointConvModel(config['train']['batch_size'], config['bn'], config['num_classes'])

    callbacks = [
        keras.callbacks.TensorBoard(
            '{}/logs'.format(config['working_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            '{}/logs/model/weights'.format(config['working_dir']), 'val_sparse_categorical_accuracy',
            save_best_only=True)
    ]

    model.build((config['train']['batch_size'], 65536, 4))  # 8192, 3
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(),]
    )

    train_ds = load_dataset(config['trainfiles'][:3], config['train']['batch_size'])
    val_ds = load_dataset(config['testfiles'], config['train']['batch_size'])

    # model.run_eagerly = True
    model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=10,
        validation_freq=1,
        callbacks=callbacks,
        epochs=10,
        verbose=1
    )


if __name__ == '__main__':
    train()
