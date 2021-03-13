# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score
from boltons.iterutils import chunked_iter
from bert4keras.snippets import sequence_padding


class Metrics(Callback):
    def __init__(self,
                 batch_size,
                 max_len,
                 eval_data,
                 min_delta=1e-4,
                 patience=10):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater

        self.batch_size = batch_size
        self.max_len = max_len
        self.eval_data = eval_data
        self.history = defaultdict(list)

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def calc_metrics(self):
        y_true, y_pred = [], []
        for chunk in chunked_iter(self.eval_data, self.batch_size):
            token_ids = [obj['token_ids'] for obj in chunk]
            segment_ids = [obj['segment_ids'] for obj in chunk]
            tcol_ids = [obj['tcol_ids'] for obj in chunk]
            true_labels = [obj['label_id'] for obj in chunk]

            token_ids = sequence_padding(token_ids, length=self.max_len)
            segment_ids = sequence_padding(segment_ids, length=self.max_len)
            tcol_ids = sequence_padding(tcol_ids)
            pred = self.model.predict([token_ids, segment_ids, tcol_ids])
            pred = np.argmax(pred, 1)
            y_true += list(true_labels)
            y_pred += list(pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1, acc

    def on_epoch_end(self, epoch, logs=None):
        val_f1, val_acc = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        print(f"- val_acc {val_acc} - val_f1 {val_f1}")
        if self.monitor_op(val_f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, val_f1):
            self.best = val_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
