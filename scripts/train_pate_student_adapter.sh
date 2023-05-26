#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

set -euo pipefail

# source ./venv3/bin/activate

lang=en
nb_teachers=50
nepochs=200
batch_size=512
eps=7.96
delta=1e-5
stage=0
teacher_folder=
prefix=
adapter_dim=192
. parse_options.sh
[ -z $prefix ] && echo "Error: Missing --prefix" && exit 1

KWS_PATH=$PWD
if [ -z $teacher_folder ]; then
    TEACHER_FOLDER=exp/${lang}_pate_teachers_${nb_teachers}_adapter_${adapter_dim}
else
    TEACHER_FOLDER=$teacher_folder
fi

EXP=exp/${lang}_pate_students_${nb_teachers}_${eps}_adapter_${adapter_dim}
DATA_PATH=$prefix/$lang
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels/kwt3
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

NUM_TRAIN=`wc -l $DATA_PATH/filtered/${lang}_test.csv | cut -d ' ' -f 1`
NUM_TRAIN=`echo ${NUM_TRAIN} - 1 | bc`
NUM_STEPS=`echo ${NUM_TRAIN} \* $nepochs / ${batch_size} / 2 | bc`

WANTED_WORD=`cut -d ' ' -f 1 $DATA_PATH/filtered/word_counts.txt | paste -sd,`
START_CHECKPOINT=$MODELS_PATH/best_weights

TEACHER_MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time
--adapter_dim $adapter_dim
--fix_transformer
"

STUDENT_MODEL_ARGS="kws_transformer 
--num_layers 12
--heads 3
--d_model 192
--mlp_dim 768
--dropout1 0.
--attention_type time
--adapter_dim $adapter_dim
--fix_transformer
"

TRAIN_ARGS="--wanted_words $WANTED_WORD
    --start_checkpoint $EXP/init.hdf5
    --lang $lang
    --train_dir $EXP/
    --mel_upper_edge_hertz 7600
    --optimizer adamw
    --lr_schedule cosine
    --how_many_training_steps $NUM_STEPS
    --data_dir $DATA_PATH/
    --eval_step_interval 20
    --warmup_epochs 10
    --l2_weight_decay 0.1
    --alsologtostderr
    --learning_rate 0.001
    --batch_size $batch_size
    --label_smoothing 0.1
    --window_size_ms 30.0
    --window_stride_ms 10.0
    --mel_num_bins 80
    --dct_num_features 40
    --resample 0.15
    --train 1
    --split 0
    --use_spec_augment 1
    --time_masks_number 2
    --time_mask_max_size 25
    --frequency_masks_number 2
    --frequency_mask_max_size 7
    --pick_deterministically 1"

mkdir -p $EXP

if [ $stage -le 0 ]; then
    if [ ! -f $TEACHER_FOLDER/teacher_preds.npy ] ; then
        echo "Stage 1: Generating teacher predictions"
        python pate/gen_teacher_pred.py $TRAIN_ARGS \
            --nb_teachers $nb_teachers \
            --dataset_class MLSW_data.MLSWProcessor \
            --pate_teacher_folder $TEACHER_FOLDER \
            $TEACHER_MODEL_ARGS
    fi
fi

if [ $stage -le 1 ]; then
    echo "Stage 2: Calculate the required lap_scale"
    python pate/analysis_syft.py --counts_file=$TEACHER_FOLDER/teacher_preds.npy \
        --delta=$delta \
        --n_chunks 2 \
        --target_eps 8.0 \
        --output_dir $EXP
fi

if [ $stage -le 2 ]; then
    echo "Stage 3: Train pate student"
    lap_scale=`cat $EXP/lap_scale`
    python MLSW/convert.py $START_CHECKPOINT $EXP/init.hdf5 $STUDENT_MODEL_ARGS
    $CMD_TRAIN $TRAIN_ARGS \
        --dataset_class PATE_data.MLSW_PATE_student \
        --lap_scale $lap_scale \
        --pate_teacher_folder $TEACHER_FOLDER \
        $STUDENT_MODEL_ARGS 2>&1 | tee $EXP/train.log
fi

