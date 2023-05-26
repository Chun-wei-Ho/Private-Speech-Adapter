import argparse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from kws_streaming.train import base_parser
from kws_streaming.models import models
from kws_streaming.models import model_flags
import kws_streaming.models.kws_transformer as kws_transformer

if __name__ == '__main__':
    parser = base_parser.base_parser()
    parser.add_argument('input_checkpoint', type=str)
    parser.add_argument('output_checkpoint', type=str)
    parser.add_argument('--input_checkpoint2', type=str, default=None)
    parser.add_argument('--save_weights_only', action='store_true', default=False)
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

    parser_kws_transformer = subparsers.add_parser('kws_transformer')
    kws_transformer.model_parameters(parser_kws_transformer)

    FLAGS = parser.parse_args()

    flags = model_flags.update_flags(FLAGS)
    flags.training = False

    model = models.MODELS[flags.model_name](flags)
    model.load_weights(flags.input_checkpoint)
    if flags.input_checkpoint2 is not None:
        model.load_weights(flags.input_checkpoint2)
    if flags.save_weights_only:
        model.save_weights(flags.output_checkpoint)
    else:
        model.save(flags.output_checkpoint)
