# AGN
Official Code for Merging Statistical Feature via Adaptive Gate for Improved Text Classification (AAAI2021)

## Prepare Data

### Dataset

|    Dataset   |                                URL                               |
|:------------:|:----------------------------------------------------------------:|
|     Subj     |      http://www.cs.cornell.edu/people/pabo/movie-review-data/     |
|    SST-1/2   |                http://nlp.stanford.edu/sentiment/                |
|     TREC     |                  https://trec.nist.gov/data.html                 |
|   AG's News  | http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles      |
| Yelp P. / F. |                   https://www.yelp.com/dataset/                  |

You first need to download datasets from official sites. Then format the data into `JSONL` style, as follows:

```json
{"label": "0", "text": "hoffman waits too long to turn his movie in an unexpected direction , and even then his tone retains a genteel , prep-school quality that feels dusty and leatherbound ."}
{"label": "1", "text": "if you 're not deeply touched by this movie , check your pulse ."}
```

Each line is a JSON object, in which two fields `text` and `label` are required.



### Pretrained Language Model

We apply the pretrained `Uncased-Bert-Base` model in this paper, you can download it by [this url](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) directly. 


## Setup Environment

We recommend you create a virtual environment to conduct experiments. 

```bash
$ python -m venv agn
$ source agn/bin/activate
```

You should install TensorFlow in terms of your environment. You can install TensorFlow-GPU if exists GPU in your device, TensorFlow-CPU otherwise. Note that we only test the code under `tensorflow<2.0`, greater versions may not be compatible. We strongly recommend `tensorflow==1.15.4`.


Next, You should install other python dependencies.

```bash
$ python -m pip install -r requirements.txt
```


## Train & Evaluate

You should first prepare a configure file to set data paths and hyperparameters.

for example:

`sst2.json`

```json
{
  "max_len": 60,
  "ae_epochs": 100,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 2e-5,
  "pretrained_model_dir": "/path/to/pretrained-bert/uncased-bert-base",
  "pretrained_model_type": "bert",
  "tcol_latent_size": 200,
  "train_path": "/path/to/SST-2/train.jsonl",
  "dev_path": "/path/to/SST-2/test.jsonl",
  "epsilon": 0.2,
  "iterations": 10,
  "verbose": 1
}
```


| Parameter             | Description                                      |
|-----------------------|--------------------------------------------------|
| max_len               | max length of input sequence                     |
| ae_epochs             | epochs to train AutoEncoder                      |
| epochs                | epochs to train classifier                       |
| batch_size            | batch size                                       |
| learning_rate         | learning rate                                    |
| pretrained_model_dir  | file directory of the pre-trained language model |
| pretrained_model_type | bert or albert                                   |
| tcol_latent_size      | latent dimension size of tcol                    |
| train_path            | data path of train set                           |
| dev_path              | data path of develop set / test set              |
| epsilon               | epsilon size of valve                            |


Then you can start to train and evaluate by following shell script.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py /path/to/config.json
```
