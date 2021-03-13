# -*- coding: utf-8 -*-

import os

import keras
import keras.layers as L
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import sparse_categorical_crossentropy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from attention import SeqSelfAttention
from bert4keras.models import build_transformer_model


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

        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
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
                             callbacks=[EarlyStopping(patience=2)],
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
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, verbose=2):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=[EarlyStopping(patience=2)],
                             validation_data=(X_test, X_test), verbose=verbose)


class NonMasking(L.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class AGN(L.Layer):
    def __init__(self, epsilon=0.1, attn_initializer=None,  **kwargs):
        super(AGN, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.attn_initializer = attn_initializer
        self.supports_masking = False

    def call(self, inputs):
        X, gi = inputs
        valve = L.Activation('sigmoid')(X)
        upper = K.cast(K.greater(valve, 0.5 + self.epsilon), K.floatx())
        lower = K.cast(K.less(valve, 0.5 - self.epsilon), K.floatx())
        enhanced = X + (1.0 - (upper + lower)) * gi
        return SeqSelfAttention(attention_activation='tanh', kernel_initializer=self.attn_initializer)(enhanced)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])


class AGNClassifier:
    """ Adaptive Gate Network
    """
    def __init__(self, config):
        self.config = config
        # load pretrained bert
        self.model = None
        self.build()

    def build(self):
        K.clear_session()
        bert = build_transformer_model(
            config_path=os.path.join(self.config['pretrained_model_dir'],
                                     f'{self.config["pretrained_model_type"]}_config.json'),
            checkpoint_path=os.path.join(self.config['pretrained_model_dir'],
                                         f'{self.config["pretrained_model_type"]}_model.ckpt'),
            model=self.config["pretrained_model_type"],
            return_keras_model=False,
        )

        # GI
        gi_in = L.Input(name="gi", shape=(self.config["tcol_latent_size"], ), dtype="float32")
        gi = gi_in

        # AGN
        X = NonMasking()(bert.model.output)
        shapes = K.int_shape(X)
        X = L.Dense(shapes[-1], activation='relu')(X)  # (B, L, D)
        gi = L.Dense(self.config['max_len'], activation='relu')(gi)  # (B, L)
        gi = L.Lambda(lambda x: K.expand_dims(x, 2))(gi)
        X = AGN(epsilon=self.config['epsilon'], attn_initializer=bert.initializer)([X, gi])
        output = L.GlobalMaxPooling1D()(X)
        output = L.Dense(self.config['output_size'], activation='softmax')(output)
        self.model = Model(inputs=(*bert.model.input, gi_in), outputs=output)
        self.model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(self.config['learning_rate']),
        )
        self.model.summary()
