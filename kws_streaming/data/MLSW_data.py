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

tf.disable_eager_execution()

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None
# pylint: enable=g-direct-tensorflow-import


from .input_data import AudioProcessor
from .input_data import MAX_NUM_WAVS_PER_CLASS, SILENCE_LABEL, SILENCE_INDEX, \
                       UNKNOWN_WORD_LABEL, UNKNOWN_WORD_INDEX, prepare_words_list, \
                       BACKGROUND_NOISE_DIR_NAME, RANDOM_SEED, MAX_ABS_INT16

class MLSWProcessor(AudioProcessor):
  def __init__(self, flags):
    self.lang = flags.lang
    self.nb_teachers = flags.nb_teachers
    self.teacher_id = flags.teacher_id
    self.half_test = flags.half_test
    super().__init__(flags)
  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage, split_data):
    raise Exception("Please use --split 0 option when using MLSW")
  def prepare_split_data_index(self, wanted_words, split_data):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`,
    where `data_dir` has to contain folders (prepared by user):
      testing
      training
      validation
      _background_noise_ - contains data which are used for adding background
      noise to training data only

    Args:
      wanted_words: Labels of the classes we want to be able to recognize.
      split_data: True - split data automatically; False - user splits the data

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)

    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index

    self.words_list = prepare_words_list(wanted_words, split_data)

    self.data_index = {'validation': [], 'testing': [], 'training': []}

    metas = {'training': os.path.join(self.data_dir, 'filtered', f'{self.lang}_train.csv'),
            'validation': os.path.join(self.data_dir, 'filtered', f'{self.lang}_dev.csv'),
            'testing': os.path.join(self.data_dir, 'filtered', f'{self.lang}_test.csv')}
    all_words = {}
    match = re.compile(".opus$")
    for set_index, meta in metas.items():
      df = pd.read_csv(meta)
      for idx, row in tqdm.tqdm(df.iterrows(), desc=f"Reading {meta}"):
        if set_index == 'training' and idx % self.nb_teachers != self.teacher_id:
          continue
        wav_path = os.path.join(self.data_dir, 'data', 'clips', row['LINK'])
        new_wav_path = match.sub('.wav', wav_path)
        word = row['WORD'].lower()
        all_words[word] = True
        if word in wanted_words_index:
          self.data_index[set_index].append({'label': word, 'file': new_wav_path})
        else:
          raise Exception('Unknown word ' + word)

      if not all_words:
        raise IOError('No .wavs found at ' + search_path)
      for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
          raise IOError('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    if self.half_test:
      testdata_for_training = self.data_index['testing'][::2]
      testdata_for_testing = self.data_index['testing'][1::2]
      self.data_index['testing'] = testdata_for_testing
      self.data_index['training'] += testdata_for_training
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

    # Prepare the rest of the result data structure.
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        raise Exception('Unknown word ' + word)

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    return self.background_data
