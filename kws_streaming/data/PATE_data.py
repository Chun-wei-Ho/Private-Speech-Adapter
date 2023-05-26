import hashlib
import math
import os.path
import pandas as pd
import random
import re
import sys
import tarfile
from absl import logging
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes
from kws_streaming.models import models

import tqdm
from tqdm.contrib.concurrent import process_map
import subprocess
import re

# pylint: disable=g-direct-tensorflow-import
# below ops are on a depreciation path in tf, so we temporarily disable pylint
# to be able to import them: TODO(rybakov) - use direct tf

from argparse import Namespace

import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from pate import aggregation

tf.disable_eager_execution()

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None
# pylint: enable=g-direct-tensorflow-import


from .input_data import MAX_NUM_WAVS_PER_CLASS, SILENCE_LABEL, SILENCE_INDEX, \
                       UNKNOWN_WORD_LABEL, UNKNOWN_WORD_INDEX, prepare_words_list, \
                       BACKGROUND_NOISE_DIR_NAME, RANDOM_SEED, MAX_ABS_INT16

from .MLSW_data import MLSWProcessor

class MLSW_PATE_student(MLSWProcessor):
    def __init__(self, flags):
      self.teacher_prediction = os.path.join(flags.pate_teacher_folder, 'teacher_preds.npy')
      self.lap_scale = flags.lap_scale
      super().__init__(flags)
    def prepare_split_data_index(self, wanted_words, split_data):
      super().prepare_split_data_index(wanted_words, split_data)

      with open(self.teacher_prediction, 'rb') as f:
        teacher_preds = np.load(f)
        stdnt_labels = aggregation.noisy_max(teacher_preds, self.lap_scale)
      index_to_word = {index: word for word, index in self.word_to_index.items()}

      test_data = self.data_index['testing']

      # rewrite data_index
      self.data_index = {'validation': [], 'testing': [], 'training': []}
      correct_count = 0
      for idx, item in enumerate(test_data):
        correct_count += int(index_to_word[stdnt_labels[idx]] == item['label'])
        if idx % 2 == 0:
          item['label'] = index_to_word[stdnt_labels[idx]]
          self.data_index['training'].append(item)
          self.data_index['validation'].append(item)
        else:
          self.data_index['testing'].append(item)
      # print(f"Student accuracy with lap_scale {self.lap_scale}: {correct_count / len(test_data)}")

class MLSW_PATE_student_ASC(MLSW_PATE_student):
  def __init__(self, flags):
    teacher_dirs = [os.path.join(flags.pate_teacher_folder, d) for d in os.listdir(flags.pate_teacher_folder)]
    self.teacher_models = [os.path.join(d, "best_weights") for d in teacher_dirs if os.path.isdir(d)]

    model = models.MODELS[flags.model_name](flags)
    model.trainable = False
    
    output = model.layers[flags.AWC_layer_index].output

    self.model = model
    self.latent_model = tf.keras.Model(inputs=model.input, outputs=output)
    self.model_idx = 0

    super().__init__(flags)
  def get_data(self, *args, **kwargs):
    data, _ = super(MLSW_PATE_student_ASC, self).get_data(*args, **kwargs)
    self.model.load_weights(self.teacher_models[self.model_idx])
    label = self.latent_model.predict(data)
    self.model_idx = (self.model_idx + 1) % len(self.teacher_models)
    return data, label
