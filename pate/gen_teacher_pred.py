from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf

import argparse
from kws_streaming.train import base_parser
from kws_streaming.models import models
from kws_streaming.layers import modes
from kws_streaming.models import model_flags
import kws_streaming.models.kws_transformer as kws_transformer
from kws_streaming.models import utils

import kws_streaming.data.MLSW_data as MLSW_data

import tqdm

if __name__ == '__main__':
    parser = base_parser.base_parser()
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

    parser_kws_transformer = subparsers.add_parser('kws_transformer')
    kws_transformer.model_parameters(parser_kws_transformer)

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed and tuple(unparsed) != ('--alsologtostderr',):
        raise ValueError('Unknown argument: {}'.format(unparsed))

    flags = model_flags.update_flags(FLAGS)
    flags.training = False

    # Start a new TensorFlow session.
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


    DatasetClass = eval(flags.dataset_class)
    audio_processor = DatasetClass(flags)

    set_size = audio_processor.set_size('testing')
    tf.keras.backend.set_learning_phase(0)
    # test_batch_size = 100
    # flags.batch_size = test_batch_size
    test_batch_size = flags.batch_size
    set_size = int(set_size / test_batch_size) * test_batch_size
    model = models.MODELS[flags.model_name](flags)

    teachers_preds_shape = (flags.nb_teachers,
                    audio_processor.set_size('testing'),
                    model.output.shape[-1])

    # Create array that will hold result
    teachers_preds = np.zeros(teachers_preds_shape, dtype=np.float32)
    # ground_truth = np.zeros(audio_processor.set_size('testing'), dtype=np.int32)
    teacher_progbar = tqdm.tqdm(range(flags.nb_teachers), desc='Generating Teacher predictions')
    for teacher_id in teacher_progbar:
        weights_path = os.path.join(flags.pate_teacher_folder, str(teacher_id),
                                    'best_weights')
        model.load_weights(weights_path).expect_partial()
        for i in range(0, set_size, test_batch_size):
            test_fingerprints, test_ground_truth = audio_processor.get_data(
                test_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)
            predictions = model.predict(test_fingerprints)
            teachers_preds[teacher_id, i:i+predictions.shape[0], :] = predictions
            # ground_truth[i:i+predictions.shape[0]] = test_ground_truth

    teacher_preds_path = os.path.join(flags.pate_teacher_folder, 'teacher_preds.npy')
    with open(teacher_preds_path, 'wb') as f:
        np.save(f, teachers_preds)
    # gt_path = os.path.join(flags.train_dir, 'ground_truth.npy')
    # with open(gt_path, 'wb') as f:
    #     np.save(f, ground_truth)