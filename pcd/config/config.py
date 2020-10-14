import os
import yaml
import re
import numpy as np

# because pycharm doesn't read the environment variables so this sets the automatically
# for quick switching between machines
from os.path import expanduser
home = expanduser("~")
if home == '/Users/dodkins':
    os.environ["WORKING_DIR"] = "/Users/dodkins"
elif home == '/home/dodkins':
    # os.environ["WORKING_DIR"] = "/mnt/data0/dodkins"
    os.environ["WORKING_DIR"] = "/work/dodkins"
else:
    print('System not recognised. Make sure $WORKING_DIR is set')

path_matcher = re.compile(r'\$\{([^}^{]+)\}')
def path_constructor(loader, node):
  ''' Extract the matched value, expand env variable, and replace the match '''
  value = node.value
  match = path_matcher.match(value)
  env_var = match.group()[2:-1]
  var = os.environ.get(env_var)
  if not var:
      print(f'No environment variable with name {env_var} found')
  return var + value[match.end():]

yaml.add_implicit_resolver('!path', path_matcher)
yaml.add_constructor('!path', path_constructor)

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

## register the tag handler
yaml.add_constructor('!join', join)

# load config.yml
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml"), 'r') as stream:
    try:
        config = yaml.load(stream, Loader=yaml.Loader)
    except yaml.YAMLError as exc:
        config = exc

# deduce astro params
for astro in ['angles', 'lods', 'contrasts', 'planet_spectra']:
    if isinstance(config['data'][astro], list):
        assert len(config['data'][astro]) == config['data']['num_indata']

    elif isinstance(config['data'][astro], int):
        config['data'][astro] = [config['data'][astro]] * config['data']['num_indata']

    elif isinstance(config['data'][astro], str):  # todo replace with regex to recognise tuples - https://stackoverflow.com/questions/39553008/how-to-read-a-python-tuple-using-pyyaml
        config['data'][astro]= config['data'][astro].replace('(', '')
        config['data'][astro]= config['data'][astro].replace(')', '')
        bounds = np.float_(config['data'][astro].split(','))
        config['data'][astro] = np.random.uniform(bounds[0], bounds[1], config['data']['num_indata'])

num_test = config['data']['num_indata'] * config['data']['test_frac']
num_test = int(num_test)
num_train = config['data']['num_indata'] - num_test
num_eval = config['data']['num_indata']

config['glados_inputfiles'] = config['mec']['glados_inputfiles']

for file, num in zip(['testfiles', 'trainfiles', 'glados_inputfiles'], [num_test, num_train, num_eval]):

    suffix, extension = config[file].split('{id}')
    config[file] = [suffix + str(l) + extension for l in range(num)]

config['mec']['glados_inputfiles'] = config.pop('glados_inputfiles')

# load run.yml
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "run.yml"), 'r') as stream:
    try:
        run = yaml.load(stream)
    except yaml.YAMLError as exc:
        run = exc

if __name__ == '__main__':
    from pprint import pprint
    pprint(config)
    print('\n')
    pprint(run)
