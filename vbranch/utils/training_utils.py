import os
import pandas as pd
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def save_results(data, dirname, filename, mode='w'):
    """Helper to save `data` dict to csv"""

    # Create folder to store csv
    csv_dir = os.path.join('results', dirname)
    if not os.path.isdir(csv_dir):
        os.system('mkdir -p ' + csv_dir)

    if mode == 'w':
        results = pd.DataFrame(data=data)
    elif mode == 'a':
        results = pd.DataFrame(data=data, index=[0])
    else:
        raise ValueError('invalid file I/O mode ("w" or "a")')

    csv_path = os.path.join(csv_dir, filename)
    results.to_csv(csv_path, mode=mode)

    return csv_path

def lr_exp_decay_scheduler(init_lr, t0, t1, decay):
    """NOTE: `episode` starts from 1"""
    def func(episode):
        if episode < t0:
            return init_lr
        lr = init_lr * np.power(decay, (episode - t0) / (t1 - t0))
        return lr
    return func
