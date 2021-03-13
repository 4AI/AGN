# -*- coding: utf-8 -*-

import os
import sys
import json
from pprint import pprint

from bert4keras.tokenizers import Tokenizer

from dataloader import DataLoader, DataGenerator
from model import AGNClassifier
from metrics import Metrics


if len(sys.argv) != 2:
    print("usage: python main.py /path/to/config")
    exit()

config_file = str(sys.argv[1])

with open(config_file, "r") as reader:
    config = json.load(reader)

print("config:")
pprint(config)


# Load tokenizer
tokenizer = Tokenizer(os.path.join(config['pretrained_model_dir'], 'vocab.txt'), do_lower_case=True)

accuracy_list = []
f1_list = []
for idx in range(1, config['iterations'] + 1):
    print("load data...")
    dataloader = DataLoader(tokenizer,
                            config['max_len'],
                            use_vae=True,
                            batch_size=config["batch_size"],
                            ae_latent_dim=config['tcol_latent_size'],
                            ae_epochs=config['ae_epochs'])
    dataloader.set_train(config['train_path'])
    dataloader.set_dev(config['dev_path'])
    print("build generator")
    generator = DataGenerator(config['batch_size'], config['max_len'])
    generator.set_dataset(dataloader.train_set)
    metrics_callback = Metrics(config['batch_size'], config['max_len'], dataloader.dev_set)
    config['output_size'] = dataloader.label_size
    model = AGNClassifier(config)
    print("start to fitting...")
    model.model.fit_generator(
            generator.__iter__(),
            steps_per_epoch=generator.steps_per_epoch,
            epochs=config['epochs'],
            callbacks=[metrics_callback],
            verbose=config['verbose']
    )

    accuracy = max(metrics_callback.history["val_acc"])
    f1 = max(metrics_callback.history["val_f1"])
    accuracy_list.append(accuracy)
    f1_list.append(f1)
    log = f"iteration {idx} accuracy: {accuracy}, f1: {f1}\n"
    print(log)

print("Average accuracy:", sum(accuracy_list) / len(accuracy_list))
print("Average f1:", sum(f1_list) / len(f1_list))
