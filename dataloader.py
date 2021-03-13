# -*- coding: utf-8 -*-

""" DataLoader
"""

import os
import re
import random
import json
from collections import defaultdict

import numpy as np
from bert4keras.snippets import sequence_padding

from model import VariationalAutoencoder, Autoencoder

SEED = 20201101
os.environ['PYTHONHASHSEED'] = f'{SEED}'
np.random.seed(SEED)
random.seed(SEED)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = string.replace("\n", "")
    string = string.replace("\t", "")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class DataGenerator:
    def __init__(self, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len

    def set_dataset(self, train_set):
        self.train_set = train_set
        self.train_size = len(self.train_set)
        self.train_steps = len(self.train_set) // self.batch_size
        if self.train_size % self.batch_size != 0:
            self.train_steps += 1

    def __iter__(self, shuffle=True):
        while True:
            idxs = list(range(self.train_size))
            if shuffle:
                np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, batch_tcol_ids, batch_label_ids = [], [], [], []
            for idx in idxs:
                d = self.train_set[idx]
                batch_token_ids.append(d['token_ids'])
                batch_segment_ids.append(d['segment_ids'])
                batch_tcol_ids.append(d['tcol_ids'])
                batch_label_ids.append(d['label_id'])
                if len(batch_token_ids) == self.batch_size or idx == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids, length=self.max_len)
                    batch_segment_ids = sequence_padding(batch_segment_ids, length=self.max_len)
                    batch_tcol_ids = sequence_padding(batch_tcol_ids)
                    batch_label_ids = np.array(batch_label_ids)
                    yield [batch_token_ids, batch_segment_ids, batch_tcol_ids], batch_label_ids
                    batch_token_ids, batch_segment_ids, batch_tcol_ids, batch_label_ids = [], [], [], []

    @property
    def steps_per_epoch(self):
        return self.train_steps


class DataLoader:
    def __init__(self, tokenizer, max_len, use_vae=False, batch_size=64, ae_latent_dim=64, ae_epochs=20):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.train_steps = 0
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tcol_info = defaultdict(dict)
        self.tcol = {}
        self.label2idx = {}
        self.token2cnt = defaultdict(int)

        self.pad = '<pad>'
        self.unk = '<unk>'

    def set_train(self, train_path):
        """set train dataset"""
        self._train_set = self._read_data(train_path, build_vocab=True)

    def set_dev(self, dev_path):
        """set dev dataset"""
        self._dev_set = self._read_data(dev_path)

    def set_test(self, test_path):
        """set test dataset"""
        self._test_set = self._read_data(test_path)

    @property
    def train_set(self):
        return self._train_set

    @property
    def dev_set(self):
        return self._dev_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def label_size(self):
        return len(self.label2idx)

    def save_dataset(self, setname, fpath):
        if setname == 'train':
            dataset = self.train_set
        elif setname == 'dev':
            dataset = self.dev_set
        elif setname == 'test':
            dataset = self.test_set
        else:
            raise ValueError(f'not support set {setname}')
        with open(fpath, 'w') as writer:
            for data in dataset:
                writer.writelines(json.dumps(data, ensure_ascii=False) + "\n")

    def load_dataset(self, setname, fpath):
        if setname not in ['train', 'dev', 'test']:
            raise ValueError(f'not support set {setname}')
        dataset = []
        with open(fpath, 'r') as reader:
            for line in reader:
                dataset.append(json.loads(line.strip()))
        if setname == 'train':
            self._train_set = dataset
        elif setname == 'dev':
            self._dev_set = dataset
        elif setname == 'test':
            self._test_set = dataset

    def add_tcol_info(self, token, label):
        """ add TCoL
        """
        if label not in self.tcol_info[token]:
            self.tcol_info[token][label] = 1
        else:
            self.tcol_info[token][label] += 1

    def set_tcol(self):
        """ set TCoL
        """
        self.tcol[0] = np.array([0] * self.label_size)  # pad
        self.tcol[1] = np.array([0] * self.label_size)  # unk
        self.tcol[0] = np.reshape(self.tcol[0], (1, -1))
        self.tcol[1] = np.reshape(self.tcol[1], (1, -1))
        for token, label_dict in self.tcol_info.items():
            vector = [0] * self.label_size
            for label_id, cnt in label_dict.items():
                vector[label_id] = cnt / self.token2cnt[token]
            vector = np.array(vector)
            self.tcol[token] = np.reshape(vector, (1, -1))

    def _read_data(self, fpath, build_vocab=False):
        datas = []
        with open(fpath, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line)
                obj['text'] = clean_str(obj['text'])
                if build_vocab:
                    if obj['label'] not in self.label2idx:
                        self.label2idx[obj['label']] = len(self.label2idx)
                token_ids, segment_ids = self.tokenizer.encode(
                    obj['text'], maxlen=self.max_len,
                )
                for token in token_ids:
                    self.token2cnt[token] += 1
                    self.add_tcol_info(token, self.label2idx[obj['label']])
                datas.append({'token_ids': token_ids, 'segment_ids': segment_ids, 'label_id': self.label2idx[obj['label']]})
            if self.use_vae:
                print("batch alignment...")
                print("previous data size:", len(datas))
                keep_size = len(datas) // self.batch_size
                datas = datas[:keep_size * self.batch_size]
                print("alignment data size:", len(datas))
            if build_vocab:
                print("set tcol....")
                self.set_tcol()
                print("token size:", len(self.tcol))
                print("done to set tcol...")
            tcol_vectors = []
            for data in datas:
                padded = [0] * (self.max_len - len(data['token_ids']))
                token_ids = data['token_ids'] + padded
                tcol_vector = np.concatenate([self.tcol.get(token, self.tcol[1]) for token in token_ids[:self.max_len]])
                tcol_vector = np.reshape(tcol_vector, (1, -1))
                tcol_vectors.append(tcol_vector)
            print("train vae...")
            X = np.concatenate(tcol_vectors)
            if self.use_vae:
                autoencoder = VariationalAutoencoder(latent_dim=self.ae_latent_dim, epochs=self.ae_epochs, batch_size=self.batch_size)
            else:
                autoencoder = Autoencoder(latent_dim=self.ae_latent_dim, epochs=self.ae_epochs, batch_size=self.batch_size)
            autoencoder.fit(X)
            X = autoencoder.encoder.predict(X, batch_size=self.batch_size)
            # decomposite
            assert len(X) == len(datas)
            for x, data in zip(X, datas):
                data['tcol_ids'] = x.tolist()
        return datas
