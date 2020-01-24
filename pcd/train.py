import os
from config.config import config
if config['task'] == 'part_seg':
    import pointnet.part_seg.train as pointnet
elif config['task'] == 'sem_seg':
    import pointnet.sem_seg.train as pointnet

def train():
    if os.path.exists(config['train']['ml_meta']):
        os.remove(config['train']['ml_meta'])
    pointnet.train()

if __name__ == '__main__':
    train()
