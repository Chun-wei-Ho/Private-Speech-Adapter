# coding=utf-8
# Copyright (c) 2021, Arm Limited and Contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train utility functions, based on tensorflow/examples/speech_commands.

  It consists of several steps:
  1. Creates model.
  2. Reads data
  3. Trains model
  4. Select the best model and evaluates it
"""

import json
from types import SimpleNamespace
import os.path
import pprint
from absl import logging
import numpy as np
import tensorflow as tf_v2
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import kws_streaming.data.input_data as input_data
import kws_streaming.data.MLSW_data as MLSW_data
import kws_streaming.data.PATE_data as PATE_data
from kws_streaming.models import models
from kws_streaming.models import utils

from .kt_loss import kt_loss

from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

import math

from transformers import AdamWeightDecay


def train(flags):
  """Model training."""

  flags.training = True

  # Set the verbosity based on flags (default is INFO, so we see all messages)
  logging.set_verbosity(flags.verbosity)

  # Start a new TensorFlow session.
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  DatasetClass = eval(flags.dataset_class)
  audio_processor = DatasetClass(flags)
  source_processor = None
  if flags.asc_type != '':
    DatasetClass = eval(flags.source_dataset_class)
    source_processor = DatasetClass(flags)

  time_shift_samples = int((flags.time_shift_ms * flags.sample_rate) / 1000)

  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, flags.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, flags.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  logging.info(flags)

  model = models.MODELS[flags.model_name](flags)
  if flags.distill_teacher_json:
    with open(flags.distill_teacher_json, 'r') as f:
      teacher_flags = json.load(f, object_hook=lambda d: SimpleNamespace(
        **{ k: v for k, v in flags.__dict__.items() if not k in d },
        **d))
    teacher_base = models.MODELS[teacher_flags.model_name](teacher_flags)
    hard_labels = tf.keras.layers.Lambda(lambda logits: tf.one_hot(tf.math.argmax(logits, axis=-1), depth=flags.label_count))
    teacher = tf.keras.models.Sequential([teacher_base, hard_labels])
    teacher_base.trainable = False
    teacher.trainable = False
  else:
    teacher = None
    teacher_flags = None

  base_model = model
  latent_model = None
  if source_processor is not None:
    latent_output = model.layers[flags.AWC_layer_index].output
    latent_input = tf.keras.layers.Input(shape=latent_output.shape)
    latent_model = \
      tf.keras.Model(inputs=model.input, outputs=[model.output, latent_output])

  logging.info(model.summary())

  # save model summary
  utils.save_model_summary(model, flags.train_dir)

  # save model and data flags
  with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
    pprint.pprint(flags, stream=f)

  reduction=tf_v2.keras.losses.Reduction.AUTO
  if flags.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon)
  elif flags.optimizer == 'momentum':
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
  elif flags.optimizer == 'novograd':
    optimizer = tfa.optimizers.NovoGrad(
        lr=0.05,
        beta_1=flags.novograd_beta_1,
        beta_2=flags.novograd_beta_2,
        weight_decay=flags.novograd_weight_decay,
        grad_averaging=bool(flags.novograd_grad_averaging))
  elif flags.optimizer == 'adamw':
    # Exclude some layers for weight decay
    exclude = ["pos_emb", "class_emb", "layer_normalization", "bias"]
    optimizer = AdamWeightDecay(learning_rate=0.05, weight_decay_rate=flags.l2_weight_decay, exclude_from_weight_decay=exclude)
  elif flags.optimizer == 'dpsgd':
    reduction = tf_v2.keras.losses.Reduction.NONE
    optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=flags.dpsgd_norm_clip,
          noise_multiplier=flags.dpsgd_noise_multiplier,
          num_microbatches=None,
          learning_rate=float(flags.learning_rate))
  else:
    raise ValueError('Unsupported optimizer:%s' % flags.optimizer)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=flags.label_smoothing, reduction=reduction)
  metrics = ['accuracy']

  loss_weights = [ 0.5, 0.5, 0.0 ] if teacher else [ 1. ] # equally weight losses form label and teacher, ignore ensemble output
  
  model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
  if latent_model is not None:
    loss2 = kt_loss[flags.asc_type]
    latent_model.compile(optimizer=optimizer, loss=[loss, loss2], \
            loss_weights=[*loss_weights, 0.5], metrics=metrics)

  train_writer = tf.summary.FileWriter(flags.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(flags.summaries_dir + '/validation')

  sess.run(tf.global_variables_initializer())

  if flags.start_checkpoint:
    model.load_weights(flags.start_checkpoint, by_name=True, skip_mismatch=True)
    logging.info('Weights loaded from %s', flags.start_checkpoint)

  if teacher_flags and teacher_flags.start_checkpoint:
    # Load weights into teacher base as this is the actual model that was saved, teacher includes hard label head
    teacher_base.load_weights(teacher_flags.start_checkpoint).assert_existing_objects_matched()
    logging.info('Distillation teacher weights loaded from %s', teacher_flags.start_checkpoint)

  start_step = 0

  logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, flags.train_dir, 'graph.pbtxt')

  # Save list of words.
  with tf.io.gfile.GFile(os.path.join(flags.train_dir, 'labels.txt'), 'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  best_accuracy = 0.0

  # prepare parameters for exp learning rate decay
  training_steps_max = np.sum(training_steps_list)
  lr_init = learning_rates_list[0]
  exp_rate = -np.log(learning_rates_list[-1] / lr_init)/training_steps_max
  mode = 'training'

  if flags.lr_schedule == 'cosine':
    # Currently, no restarts are performed, so it is just a cosine decay over the entire
    # training process. I think this is how DeiT does it.
    lr_init = lr_init * flags.batch_size / 512
    num_train = audio_processor.set_size(mode)
    warmup_steps = int((num_train / flags.batch_size) * flags.warmup_epochs)
    first_decay_steps=training_steps_max

  num_train_samples = len(audio_processor.data_index['training'])
  # Training loop.
  for training_step in range(start_step, training_steps_max + 1):
    if training_step > 0:
      offset = (training_step -
                1) * flags.batch_size if flags.pick_deterministically else 0

      # Pull the audio samples we'll use for training.
      train_fingerprints, train_ground_truth = audio_processor.get_data(
          flags.batch_size, offset, flags, flags.background_frequency,
          flags.background_volume, time_shift_samples, mode,
          flags.resample, flags.volume_resample, sess)
      if source_processor is not None:
        source_fingerprints, source_latent = source_processor.get_data(
            flags.batch_size, offset, flags, flags.background_frequency,
            flags.background_volume, time_shift_samples, mode,
            flags.resample, flags.volume_resample, sess)

      if flags.lr_schedule == 'exp':
        learning_rate_value = lr_init * np.exp(-exp_rate * training_step)
      elif flags.lr_schedule == 'linear':
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
          training_steps_sum += training_steps_list[i]
          if training_step <= training_steps_sum:
            learning_rate_value = learning_rates_list[i]
            break
      elif flags.lr_schedule == 'cosine':
        learning_rate_value = lr_init * min(1, float(training_step) / max(1, warmup_steps)) * (math.cos(math.pi * training_step / training_steps_max) + 1) / 2.
      else:
        raise ValueError('Wrong lr_schedule: %s' % flags.lr_schedule)

      if flags.optimizer != 'dpsgd':
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate_value)
      else:
        learning_rate_value = float(flags.learning_rate)

      one_hot_labels = tf.keras.utils.to_categorical(train_ground_truth, num_classes=flags.label_count)

      if teacher:
        teacher_labels = teacher.predict_on_batch(train_fingerprints)
        one_hot_labels = [ one_hot_labels, teacher_labels, one_hot_labels ] # third is for the ensemble output, gradient is unused

      if latent_model is not None:
        result = latent_model.train_on_batch(train_fingerprints, [one_hot_labels, source_latent])
      else:
        result = model.train_on_batch(train_fingerprints, one_hot_labels)

      if teacher:
        loss_total, loss_label, loss_teacher, loss_average, acc_label, acc_teacher, acc_ensemble = result
        differences = (teacher_labels != one_hot_labels).astype(dtype=int).sum()
        logging.info(
            'Step #%d: rate %f, accuracy %.2f%%, cross entropy %f, teacher acc %.2f%% (%d diff), teacher cross entropy %f, ensemble acc %.2f%%',
            *(training_step, learning_rate_value, acc_label * 100, loss_total, acc_teacher * 100, differences, loss_teacher, acc_ensemble * 100))
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=acc_label),
            tf.Summary.Value(tag='teacher_accuracy', simple_value=acc_teacher),
            tf.Summary.Value(tag='ensemble_accuracy', simple_value=acc_ensemble),
        ])
      else:
        if latent_model is not None:
          loss_label, _, loss_average, acc_label, _ = result
          loss_label = np.mean(loss_label)
          logging.info(
              'Step #%d: rate %f, accuracy %.2f%%, cross entropy %f',
              *(training_step, learning_rate_value, acc_label * 100, loss_label))
        else:
          loss_label, acc_label = result
          loss_label = np.mean(loss_label)
          logging.info(
              'Step #%d: rate %f, accuracy %.2f%%, cross entropy %f',
              *(training_step, learning_rate_value, acc_label * 100, loss_label)) 

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=acc_label),
        ])

      train_writer.add_summary(summary, training_step)
    is_last_step = (training_step == training_steps_max)
    if flags.optimizer=='dpsgd' and \
      ((training_step * flags.batch_size) % num_train_samples < flags.batch_size or is_last_step):
      epoch = (training_step * flags.batch_size) // num_train_samples
      if is_last_step: epoch += int((training_step * flags.batch_size) % num_train_samples != 0)
      if epoch != 0:
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
              num_train_samples, flags.batch_size, flags.dpsgd_noise_multiplier, epoch, flags.dpsgd_delta)
        logging.info('Epoch %d: For delta = %.2e, eps = %.2f' % (epoch, flags.dpsgd_delta, eps))
        summary = tf.Summary(value=[
          tf.Summary.Value(tag='epsilon', simple_value=eps),
        ])
        train_writer.add_summary(summary, training_step)
    if (training_step % flags.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      set_size = int(set_size / flags.batch_size) * flags.batch_size
      total_accuracy = 0.0
      count = 0.0
      for i in range(0, set_size, flags.batch_size):
        validation_fingerprints, validation_ground_truth = audio_processor.get_data(
            flags.batch_size, i, flags, 0.0,
            0.0, 0, 'validation',
            0.0, 0.0, sess)

        one_hot_labels = tf.keras.utils.to_categorical(validation_ground_truth, num_classes=flags.label_count)
        if teacher:
          one_hot_labels = [ one_hot_labels, one_hot_labels, one_hot_labels ]
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        result = model.test_on_batch(validation_fingerprints,
                                     one_hot_labels)

        if teacher:
          loss_total, loss_label, loss_teacher, loss_average, acc_label, acc_teacher, acc_ensemble = result
          summary = tf.Summary(value=[
          tf.Summary.Value(tag='accuracy', simple_value=acc_ensemble),
          tf.Summary.Value(tag='label_head_accuracy', simple_value=acc_label),
          tf.Summary.Value(tag='distill_head_accuracy', simple_value=acc_teacher),
          ])
          accuracy = acc_ensemble
        else:
          loss_label, acc_label = result
          summary = tf.Summary(value=[
              tf.Summary.Value(tag='accuracy', simple_value=acc_label),])
          accuracy = acc_label

        validation_writer.add_summary(summary, training_step)

        total_accuracy += accuracy
        count = count + 1.0

      total_accuracy = total_accuracy / count
      logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)',
                   *(training_step, total_accuracy * 100, set_size))

      # Save the model checkpoint when validation accuracy improves
      if total_accuracy >= best_accuracy:
        best_accuracy = total_accuracy
        # overwrite the best model weights
        model.save_weights(flags.train_dir + 'best_weights')
      logging.info('So far the best validation accuracy is %.2f%%',
                   (best_accuracy * 100))

  tf.keras.backend.set_learning_phase(0)
  set_size = audio_processor.set_size('testing')
  set_size = int(set_size / flags.batch_size) * flags.batch_size
  logging.info('set_size=%d', set_size)
  total_accuracy = 0.0
  count = 0.0

  for i in range(0, set_size, flags.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        flags.batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    one_hot_labels = tf.keras.utils.to_categorical(test_ground_truth, num_classes=flags.label_count)
    if teacher:
      one_hot_labels = [ one_hot_labels, one_hot_labels, one_hot_labels ]
    result = model.test_on_batch(test_fingerprints, one_hot_labels)

    total_accuracy += result[-1] if teacher else result[1]
    count = count + 1.0
  total_accuracy = total_accuracy / count

  logging.info('Final test accuracy = %.2f%% (N=%d)',
               *(total_accuracy * 100, set_size))
  with open(os.path.join(flags.train_dir, 'accuracy_last.txt'), 'wt') as fd:
    fd.write(str(total_accuracy * 100))
  model.save_weights(flags.train_dir + 'last_weights')
