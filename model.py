# -*- coding: utf-8 -*-

import os

seed_value = int(os.getenv('RANDOM_SEED', -1))
if seed_value != -1:
    import random
    random.seed(seed_value)
    import numpy as np
    np.random.seed(seed_value)
    import tensorflow as tf
    tf.set_random_seed(seed_value)

from langml import keras, K, L
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from langml.plm.bert import load_bert
from langml.layers import SelfAttention
from bert4keras.optimizers import Adam, extend_with_weight_decay, extend_with_piecewise_linear_lr


def search_layer(inputs, name, exclude_from=None):
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def fgm(model, embedding_name, epsilon=1):
    # modified from: https://github.com/bojone/bert4keras/blob/master/examples/task_iflytek_adversarial_training.py

    if model.train_function is None:
        model._make_train_function()

    old_train_function = model.train_function

    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    embeddings = embedding_layer.embeddings
    gradients = K.gradients(model.total_loss, [embeddings])
    gradients = K.zeros_like(embeddings) + gradients[0]

    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )

    def train_function(inputs):
        grads = embedding_gradients(inputs)[0]
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)
        K.set_value(embeddings, K.eval(embeddings) + delta)
        outputs = old_train_function(inputs)
        K.set_value(embeddings, K.eval(embeddings) - delta)
        return outputs

    model.train_function = train_function


class Sampling(L.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.int_shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder:
    def __init__(self, latent_dim=64, hidden_dim=128, activation='relu', epochs=10, batch_size=64):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = L.Input(batch_shape=(self.batch_size, input_dim))
        hidden = L.Dense(self.hidden_dim, activation=self.activation)(input_vec)
        z_mean = L.Dense(self.latent_dim)(hidden)
        z_log_var = L.Dense(self.latent_dim)(hidden)
        encoded = Sampling()([z_mean, z_log_var])
        decoded = L.Dense(input_dim, activation="sigmoid")(encoded)

        # custom loss
        def vae_loss(y_true, y_pred):
            reconstruction_loss = K.mean(
                keras.losses.binary_crossentropy(y_true, y_pred)
            )
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.mean(kl_loss)
            kl_loss *= -0.5
            return reconstruction_loss + kl_loss

        self.autoencoder = keras.Model(input_vec, decoded)
        self.encoder = keras.Model(input_vec, encoded)
        self.autoencoder.compile(optimizer='adam', loss=vae_loss)

    def fit(self, X, verbose=2):
        if not self.autoencoder:
            self._compile(X.shape[1])
        per_size = (len(X) * 0.9) // self.batch_size
        train_size = int((per_size + 1) * self.batch_size)
        X_shuffle = shuffle(X)
        X_train = X_shuffle[:train_size]
        X_test = X_shuffle[train_size:]
        print("train size:", len(X_train))
        print("dev size:", len(X_test))
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                             validation_data=(X_test, X_test), verbose=verbose)


class Autoencoder:
    def __init__(self, latent_dim=64, activation='relu', epochs=10, batch_size=64):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = L.Input(shape=(input_dim,))
        encoded = L.Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = L.Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = keras.Model(input_vec, decoded)
        self.encoder = keras.Model(input_vec, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, verbose=2):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                             validation_data=(X_test, X_test), verbose=verbose)


class AGN(L.Layer):
    def __init__(self, epsilon=0.1, **kwargs):
        super(AGN, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.supports_masking = False

    def call(self, inputs):
        X, gi = inputs
        fea_dim = K.int_shape(X)[-1]
        valve = L.Dense(fea_dim, activation='sigmoid')(X)
        X_t = L.Dense(fea_dim, activation='relu')(X)
        upper = K.cast(K.greater(valve, 0.5 + self.epsilon), K.floatx())
        lower = K.cast(K.less(valve, 0.5 - self.epsilon), K.floatx())
        enhanced = X_t + (1.0 - (upper + lower)) * gi
        return SelfAttention(return_attention=True)(enhanced)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1], input_shape[0][2]),
                (input_shape[0][0], input_shape[0][1], input_shape[0][1])]

    def compute_mask(self, inputs, mask=None):
        return [mask, None]


class AGNClassifier:
    """ Adaptive Gate Network
    """
    def __init__(self, config):
        self.config = config
        # load pretrained bert
        self.model = None
        self.attn_model = None
        self.build()

    def build(self):
        bert_model, _ = load_bert(
            config_path=os.path.join(self.config['pretrained_model_dir'],
                                     'bert_config.json'),
            checkpoint_path=os.path.join(self.config['pretrained_model_dir'],
                                         'bert_model.ckpt'),
        )
        text_mask = L.Lambda(lambda x: K.cast(
            K.expand_dims(K.greater(x, 0), 2), K.floatx()))(bert_model.input[0])
        # GI
        gi_in = L.Input(name="gi", shape=(self.config["max_len"], ), dtype="float32")
        gi = gi_in

        # AGN
        X = bert_model.output
        gi = L.Dense(self.config['max_len'], activation='tanh')(gi)  # (B, L)
        gi = L.Lambda(lambda x: K.expand_dims(x, 2))(gi)  # (B, L, 1)
        X, attn_weight = AGN(epsilon=self.config['epsilon'])([X, gi])
        X = L.Lambda(lambda x:  x[0] - 1e10 * (1.0 - x[1]))([X, text_mask])
        output = L.Lambda(lambda x: K.max(x, 1))(X)
        #output = L.Dense(128, activation='relu')(output)
        output = L.Dropout(self.config.get('dropout', 0.2))(output)
        output = L.Dense(self.config['output_size'], activation='softmax')(output)
        self.model = keras.Model(inputs=(*bert_model.input, gi_in), outputs=output)
        self.attn_model = keras.Model(inputs=(*bert_model.input, gi_in), outputs=attn_weight)

        optimizer = extend_with_weight_decay(Adam)
        optimizer = extend_with_piecewise_linear_lr(optimizer)
        optimizer_params = {
            'learning_rate': self.config['learning_rate'],
            'lr_schedule': {
                self.config['steps_per_epoch'] * 2: 1,
                self.config['steps_per_epoch'] * 3: 0.2,
                self.config['steps_per_epoch'] * self.config['epochs']: 0.1
            },
            'weight_decay_rate': 0.01,
            'exclude_from_weight_decay': ['Norm', 'bias'],
            'bias_correction': False,
        }

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer(**optimizer_params),
        )
        self.model.summary()

        if self.config.get('apply_fgm', True):
            print('apply fgm')
            fgm(self.model, 'Embedding-Token', self.config.get('fgm_epsilon', 0.2))
