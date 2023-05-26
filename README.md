# Train KT with Differential Privacy

<img src="kwt.png" alt="drawing" width="200"/>

This is the official repository for the paper [Differentially Private Adapters for Parameter Efficient Acoustic Modeling](https://arxiv.org/abs/2305.11360), presented at Interspeech 2023. We use the [code implemented by the ARM lab](https://github.com/ARM-software/keyword-transformer). Consider citing our paper and their paper if you find this work useful.

```
@article{ho2023differentially,
  title={Differentially Private Adapters for Parameter Efficient Acoustic Modeling},
  author={Ho, Chun-Wei and Yang, Chao-Han Huck and Siniscalchi, Sabato Marco},
  journal={arXiv preprint arXiv:2305.11360},
  year={2023}
}

@inproceedings{berg21_interspeech,
  author={Axel Berg and Mark O’Connor and Miguel Tairum Cruz},
  title={{Keyword Transformer: A Self-Attention Model for Keyword Spotting}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4249--4253},
  doi={10.21437/Interspeech.2021-1286}
}
```


## Setup


### Install dependencies

Set up a new virtual environment:

```shell
pip install virtualenv
virtualenv --system-site-packages -p python3 ./venv3
source ./venv3/bin/activate
```

To install dependencies, run

```shell
pip install -r requirements.txt
```

Tested using Tensorflow 2.4.0rc1 with CUDA 11.

**Note**: Installing the correct Tensorflow version is important for reproducibility! Using more recent versions of Tensorflow results in small accuracy differences each time the model is evaluated. This might be due to a change in how the random seed generator is implemented, and therefore changes the sampling of the "unknown"  keyword class.

### Data preparation
1. Download the [MLSW dataset](https://mlcommons.org/en/multilingual-spoken-words/)
2. Create subsets of MLSW dataset using the following commands
```
./MLSW/filter_dataset.sh --prefix <MLSW PATH>
```

## Model
The Keyword-Transformer model is defined [here](kws_streaming/models/kws_transformer.py). It takes the mel scale spectrogram as input, which has shape 98 x 40 using the default settings, corresponding to the 98 time windows with 40 frequency coefficients.

There are three variants of the Keyword-Transformer model:

* **Time-domain attention**: each time-window is treated as a patch, self-attention is computed between time-windows
* **Frequency-domain attention**: each frequency is treated as a patch self-attention is computed between frequencies
* **Combination of both**: The signal is fed into both a time- and a frequency-domain transformer and the outputs are combined
* **Patch-wise attention**: Similar to the vision transformer, it extracts rectangular patches from the spectrogram, so attention happens both in the time and frequency domain simultaneously.

## PATE
The original PATE method is introduced [here](https://arxiv.org/abs/1610.05755). It trains several teacher models on different chunks of the sensitive data. Then, it trains the student model on some public data and the pseudo label generate by the privately aggregated teacher model.

We made several modifications to the PATE method.
* **Knowledge Transfering**: we use a large pre-trained model to improve the performance, where we assume that there's no privacy issue in the pre-trained model.
* **Parameter Efficient Tuninig**: we use [residual adapters](https://aclanthology.org/2021.emnlp-main.541.pdf) to parameter efficiently fine-tune the model while guaranteeing DP.


## Train [DPSGD](https://ieeexplore.ieee.org/abstract/document/6736861?casa_token=HZzqxBL49VQAAAAA:j8PqgNzLomsamEhF7FnXY01LOpOVgooWfzzphwIDJFur2Fjgu4UG8lWjhDCAko9ARaTZiP3iFg)
```
./scripts/train_dpsgd.sh --prefix <MLSW PATH> [options]
Options:
  --lang <en|fr|de|ru>               Default: en
  --eps <eps>                        Default: 8
  --delta <delta>                    Default: 1e-5
  --clip_norm <clip_norm>            Default: 20 (See the original DPSGD paper)
  --nepochs <nepochs>                Default: 20
  --batch_size <batch_size>          Default: 512
```

## Train [PATE](https://arxiv.org/abs/1610.05755)
For additional option, please refer to [train_pate_student_adapter.sh](/scripts/train_pate_student_adapter.sh)
```
./scripts/train_pate_teachers_adapter.sh --prefix <MLSW path> [Options]
./scripts/train_pate_student_adpater.sh --prefix <MLSW path> [Options]
Options:
  --lang <en|fr|de|ru>                    Default: en
  --nb_teachers <number of teachers>      Default: 50
  --nepochs <nepochs>                     Default: 80
  --batch_size <batch_size>               Default: 512
  --adapter_dim <adapter dimension>       Default: 192 (Please refer to https://aclanthology.org/2021.emnlp-main.541.pdf)
```

## Acknowledgements

The code heavily borrows from the [KWS streaming work](https://github.com/google-research/google-research/tree/master/kws_streaming) by Google Research and [Keyword transformer](https://github.com/ARM-software/keyword-transformer) by the ARM lab. For a more detailed description of the code structure, see the original authors' [README](kws_streaming/README.md).

We also exploit training techniques from [DeiT](https://github.com/facebookresearch/deit).

We thank the authors for sharing their code. Please consider citing them as well if you use our code.

## License

The source files in this repository are released under the [Apache 2.0](LICENSE.txt) license.

Some source files are derived from the [KWS streaming repository](https://github.com/google-research/google-research/tree/master/kws_streaming) by Google Research. These are also released under the Apache 2.0 license, the text of which can be seen in the LICENSE file on their repository.

Some source files are derived from the [Keyword transformer](https://github.com/ARM-software/keyword-transformer) by the ARM lab. These are also released under the Apache 2.0 license, the text of which can be seen in the LICENSE file on their repository.
