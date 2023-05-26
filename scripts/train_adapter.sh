#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

set -euo pipefail

# source ./venv3/bin/activate

lang=en
adapter_dim=192
nepochs=20
batch_size=512
prefix=
. parse_options.sh

[ -z $prefix ] && echo "Error: Missing --prefix" && exit 1

KWS_PATH=$PWD
DATA_PATH=$prefix/$lang
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels/kwt3
EXP=exp/${lang}_adapter_${adapter_dim}
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

NUM_TRAIN=`wc -l $DATA_PATH/filtered/${lang}_train.csv | cut -d ' ' -f 1`
NUM_TRAIN=`echo ${NUM_TRAIN} - 1 | bc`
NUM_STEPS=`echo ${NUM_TRAIN} \* $nepochs / ${batch_size} | bc`

WANTED_WORD=`cut -d ' ' -f 1 $DATA_PATH/filtered/word_counts.txt | paste -sd,`
START_CHECKPOINT=$MODELS_PATH/best_weights

MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time
--adapter_dim $adapter_dim
--fix_transformer"

python MLSW/convert.py $START_CHECKPOINT $EXP/init.hdf5 $MODEL_ARGS

$CMD_TRAIN \
--wanted_words $WANTED_WORD \
--dataset_class 'MLSW_data.MLSWProcessor' \
--start_checkpoint $EXP/init.hdf5 \
--lang $lang \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $EXP/ \
--mel_upper_edge_hertz 7600 \
--optimizer 'adamw' \
--lr_schedule 'cosine' \
--how_many_training_steps $NUM_STEPS \
--eval_step_interval 72 \
--warmup_epochs 10 \
--l2_weight_decay 0.1 \
--alsologtostderr \
--learning_rate '0.001' \
--batch_size ${batch_size} \
--label_smoothing 0.1 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--train 1 \
--split 0 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
$MODEL_ARGS 2>&1 | tee $EXP/train.log

