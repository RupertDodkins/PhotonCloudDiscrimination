import os
import yaml

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

## register the tag handler
yaml.add_constructor('!join', join)

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml"), 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        config = exc


print(config)