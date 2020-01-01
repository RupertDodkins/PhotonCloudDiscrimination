#!/mnt/data0/miniconda/envs/medis/bin/python

import argparse
import photons
from pointnet_mod.part_seg import train
import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Photon Cloud Discrimination')
    parser.add_argument('--make-input', default=False, dest='make_input', action='store_true',
                        help='Make input data for training and testing')

    parser.add_argument('--train', default=False, dest='train', action='store_true',
                        help='Train neural network')

    parser.add_argument('--eval', default=False, dest='evaluate', action='store_true',
                        help='Apply neural network algorithm')

    args = parser.parse_args()
    if args.make_input:
        photons.make_data()
    elif args.train:
        train.train()
    elif args.evaluate:
        evaluate.evaluate()

