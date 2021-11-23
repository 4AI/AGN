# -*- coding: utf-8 -*-

""" Visualize 2D attention
"""

import os
import sys
import json
from pprint import pprint

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from langml.tokenizer import WPTokenizer

from dataloader import DataLoader
from model import AGNClassifier


if len(sys.argv) != 2:
    print("usage: python visualize_attn.py /path/to/config")
    exit()

config_file = str(sys.argv[1])

with open(config_file, "r") as reader:
    config = json.load(reader)

print("config:")
pprint(config)

# Load tokenizer
tokenizer = WPTokenizer(os.path.join(config['pretrained_model_dir'], 'vocab.txt'), lowercase=True)
tokenizer.enable_truncation(max_length=config['max_len'])

dataloader = DataLoader(tokenizer,
                        config['max_len'],
                        use_vae=True,
                        batch_size=1,
                        ae_epochs=config['ae_epochs'])

dataloader.load_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))
dataloader.load_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
config['output_size'] = dataloader.label_size
classifier = AGNClassifier(config)
classifier.model.load_weights(os.path.join(config['save_dir'], 'clf_model.weights'))


text = input('input a text: ')
text = text.replace(',', '').replace('.', '')
tokenized = tokenizer.encode(text)
token_ids = tokenized.ids[:config['max_len']] + [0] * (config['max_len'] - len(tokenized.ids))
segment_ids = [0] * len(token_ids)
data = [{'token_ids': token_ids, 'segment_ids': segment_ids}]
data = dataloader.parse_tcol_ids(data)
token_ids = np.array([data[0]['token_ids']])
segment_ids = np.array([data[0]['segment_ids']])
tcol_ids = np.array([data[0]['tcol_ids']])
logits = classifier.attn_model.predict([token_ids, segment_ids, tcol_ids])
logits = logits[0][:len(tokenized.tokens)]
# visualize
ax, fig = plt.subplots(figsize=[20, 8])
ax = sns.heatmap(logits[1:-1], linewidth=1)
ax.set_yticklabels(tokenized.tokens[1:-1])

plt.show()
plt.savefig('attn_visualize.png')
