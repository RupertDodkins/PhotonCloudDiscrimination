#!/mnt/data0/miniconda/envs/medis/bin/python

import os
# import argparse
import data
import train
import evaluate
import predict
from pcd.config.config import config, run
from multiprocessing import Pool
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
            choice = input(f"Overwrite {config['working_dir']} ? [Y/n]")
            if choice == 'n':
                pass
        else:
            data.make_input(config)

    if run['train'] and run['evaluate']['metric_funcs']:
        pool = Pool(processes=len(run['evaluate']['metric_funcs']) + 1)
        pool.apply_async(train.train, [config])
        run_multiple_func(pool, run['evaluate']['metric_funcs'])
        # funcs = []
        # for func in run['evaluate']['metric_funcs']:
        #     evaluate_func = getattr(evaluate, func)
        #     pool.apply_async(evaluate_func, )
        # 
        # pool.close()
        # pool.join()
        # 
        # for func in funcs:
        #     func.get()

    elif run['train']:
        train.train(config)

    elif run['evaluate']['metric_funcs']:
        pool = Pool(processes=len(run['evaluate']['metric_funcs']))
        run_multiple_func(pool, run['evaluate']['metric_funcs'])
        # funcs = []
        # for func in run['evaluate']['metric_funcs']:
        #     evaluate_func = getattr(evaluate, func)
        #     funcs.append(pool.apply_async(evaluate_func, ))
        #
        # pool.close()
        # pool.join()
        #
        # for func in funcs:
        #     func.get()

    if run['predict']:
        predict.predict()
