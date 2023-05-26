import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

import argparse
import numpy as np


def get_eps_args(args):
    def get_eps(multiplier):
        return compute_dp_sgd_privacy_lib \
                .compute_dp_sgd_privacy(args.num_train, args.batch_size, multiplier, args.nepochs, args.delta)[0]
    return get_eps
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("num_train", type=int)
    parser.add_argument("eps", type=float)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--nepochs", type=int, default=20)
    args = parser.parse_args()

    get_eps = get_eps_args(args)

    lower = 0
    higher = np.inf
    sys.stdout = open(os.devnull, 'w')
    while higher - lower > 1e-4:
        if higher == np.inf:
            mid = lower * 2 + 1
        else:
            mid = (lower + higher) * 0.5
        if get_eps(mid) <= args.eps:
            higher = mid
        else:
            lower = mid
    sys.stdout = sys.__stdout__

    print(higher)
