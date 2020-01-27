import os
from config.config import config
if config['task'] == 'part_seg':
    import pointnet.part_seg.train as pointnet
elif config['task'] == 'sem_seg':
    import pointnet.sem_seg.train as pointnet

def train():
    if os.path.exists(config['train']['outputs']):
        if config['overwrite_cache']:
            os.remove(config['train']['outputs'])

    pointnet.train()

if __name__ == '__main__':
    train()
