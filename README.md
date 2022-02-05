# AGN
Official Code for [Merging Statistical Feature via Adaptive Gate for Improved Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17569) (AAAI2021)

## Prepare Data

### Dataset

|    Dataset   |                                URL                               |
|:------------:|:----------------------------------------------------------------:|
|     Subj     |      http://www.cs.cornell.edu/people/pabo/movie-review-data/     |
|    SST-1/2   |                http://nlp.stanford.edu/sentiment/                |
|     TREC     |                  https://trec.nist.gov/data.html                 |
|   AG's News  | http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles      |
| Yelp P. / F. |                   https://www.yelp.com/dataset/                  |

You first need to download datasets from official sites. Then convert the data into `JSONL` style, as follows:

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

You should install TensorFlow in terms of your environment. Note that we only test the code under `tensorflow<2.0`, greater versions may not be compatible.

Our environments:

```bash
$ pip list | egrep "tensorflow|Keras|langml"
Keras                            2.3.1
langml                           0.1.0
tensorflow                       1.15.0
```

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
  "max_len": 80,
  "ae_epochs": 100,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.00003,
  "pretrained_model_dir": "/path/to/pretrained-bert/uncased-bert-base",
  "train_path": "/path/to/SST-2/train.jsonl",
  "dev_path": "/path/to/SST-2/test.jsonl",
  "save_dir": "/dir/to/save",
  "epsilon": 0.05,
  "dropout": 0.3,
  "fgm_epsilon": 0.3,
  "iterations": 1,
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
| save_dir              | dir to save model                                |
| train_path            | data path of train set                           |
| dev_path              | data path of develop set / test set              |
| epsilon               | epsilon size of valve                            |
| apply_fgm             | whether to apply fgm attack, default tru         |
| fgm_epsilon           | epsilon of fgm, default 0.2                      |


Then you can start to train and evaluate by following shell script.

```bash
export TF_KERAS=1; CUDA_VISIBLE_DEVICES=0 python main.py /path/to/config.json
```

please set `TF_KERAS=1` to use AdamW.


After training is done, models will be stored in the specified `save_dir` folder.

## Visualize Attention

To visualize attention, you should train a model first following the above instruction, then run `visualize_attn.py` as follows:

```bash
export TF_KERAS=1; CUDA_VISIBLE_DEVICES=0 python visualize_attn.py /path/to/your_config.json
```

After inputting the text to the prompt box, the code will analyze the text and save the attention figure to `attn_visualize.png`.

Note that, in previous settings, we pick up a most distinguished feature dimension from 2D attention and visualize the selected feature (1D) attention. In the latest version, we visualize the whole 2D attention rather than 1D attention.
