#!/mnt/data0/miniconda/envs/medis/bin/python

import os
# import argparse
import data
import train
import evaluate
import predict
from pcd.config.config import config, run
from multiprocessing import Pool, Process
from pprint import pprint

def run_multiple_func(pool, funcnames):
    funcs = []
    for func in funcnames:
        evaluate_func = getattr(evaluate, func)
        funcs.append(pool.apply_async(evaluate_func, ))

    pool.close()
    pool.join()

    for func in funcs:
        func.get()

if __name__ == "__main__":
    pprint(run)

    if run['new_input']:
        if os.path.exists(config['working_dir']):
            if config['overwrite_cache']:
                data.make_input(config)

    if run['train'] and run['evaluate']['metric_funcs']:
        if os.path.exists(config['train']['ml_meta']):
            if config['overwrite_cache']:
                os.remove(config['train']['ml_meta'])

        funcs = []
        funcs.append(Process(target=train.train, args=()))

        for func in run['evaluate']['metric_funcs']:
            evaluate_func = getattr(evaluate, func)
            funcs.append(Process(target=evaluate_func, args=()))

        [p.start() for p in funcs]
        [p.join() for p in funcs]

    elif run['train']:
        train.train()

    elif run['evaluate']['metric_funcs']:
        funcs = []
        for func in run['evaluate']['metric_funcs']:
            evaluate_func = getattr(evaluate, func)
            funcs.append(Process(target=evaluate_func, args=()))

        [p.start() for p in funcs]
        [p.join() for p in funcs]

    if run['predict']:
        predict.predict()
