import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model, metrics
import numpy as np
import matplotlib.pyplot as plt

class VAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(latent_dim + latent_dim)
        ])
        self.decoder = Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(center=False,scale=False),
            layers.Dense(input_dim, activation=None) 
        ])
        

    def encode(self, x):
        mean_log_var = self.encoder(x)
        mean, log_var = tf.split(mean_log_var, num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=False):
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var, z

class Discriminator(Model):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.feature_layers = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ])
        self.classifier = layers.Dense(1, activation = 'sigmoid')

    def call(self, inputs, training=False):
        features = self.feature_layers(inputs, training=training)
        output = self.classifier(features)
        return output
    
    def get_features(self, inputs, layer_idx=-1, training=False):
        if layer_idx == -1:
            return self.feature_layers(inputs, training=training)
        else:
            features = inputs
            for layer in self.feature_layers.layers[:layer_idx + 1]:
                features = layer(features, training=training)
            return features

class VAEGAN(Model):
    def __init__(self, input_dim, latent_dim, feature_layer_idx):
        super(VAEGAN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_layer_idx = feature_layer_idx

        self.vae = VAE(input_dim, latent_dim)
        self.discriminator = Discriminator(input_dim)

        self.vae_loss_tracker = metrics.Mean(name='vae_loss')
        self.reconstruction_loss_tracker = metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = metrics.Mean(name='kl_loss')
        self.gan_loss_tracker = metrics.Mean(name='gan_loss')
        self.discriminator_loss_tracker = metrics.Mean(name='discriminator_loss')

    @property
    def metrics(self):
        return [
            self.vae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.gan_loss_tracker,
            self.discriminator_loss_tracker
        ]

    def compile(self, enc_optimizer, dec_optimizer, discriminator_optimizer):
        super(VAEGAN, self).compile()
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    @tf.function
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        z_samples = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as dis_tape:
            reconstructed, mean, log_var, z = self.vae(data, training=True)
            generated = self.vae.decode(z_samples)
            encoded_generated = self.vae.decode(self.vae.encode(data)[0])

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated, training=True)
            encoded_output = self.discriminator(encoded_generated, training=True)

            real_labels = tf.ones_like(real_output) 
            fake_labels = tf.zeros_like(fake_output) 

            disc_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_output))
            disc_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, fake_output))
            disc_loss_encoded = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, encoded_output))
            discriminator_loss = disc_loss_real + disc_loss_fake + disc_loss_encoded

        dis_grads = dis_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(dis_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as enc_tape:
            reconstructed, mean, log_var, z = self.vae(data, training=True)

            real_features = self.discriminator.get_features(data, self.feature_layer_idx, training=True)
            recon_features = self.discriminator.get_features(reconstructed, self.feature_layer_idx, training=True)
            recon_loss = tf.reduce_mean(tf.square(real_features - recon_features))

            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))

            beta = 1
            encoder_loss = beta * kl_loss + recon_loss

        enc_grads = enc_tape.gradient(encoder_loss, self.vae.encoder.trainable_variables)
        self.enc_optimizer.apply_gradients(zip(enc_grads, self.vae.encoder.trainable_variables))

        with tf.GradientTape() as dec_tape:
            reconstructed, mean, log_var, z = self.vae(data, training=True)
            generated = self.vae.decode(z_samples)
            encoded_generated = self.vae.decode(self.vae.encode(data)[0])

            real_features = self.discriminator.get_features(data, self.feature_layer_idx, training=True)
            recon_features = self.discriminator.get_features(reconstructed, self.feature_layer_idx, training=True)
            recon_loss = tf.reduce_mean(tf.square(real_features - recon_features))

            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))

            real_labels = tf.ones_like(real_output) 
            fake_labels = tf.zeros_like(fake_output) 

            disc_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_output))
            disc_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, fake_output))
            disc_loss_encoded = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, encoded_output))
            discriminator_loss = disc_loss_real + disc_loss_fake + disc_loss_encoded

            gamma = 1
            gan_loss = gamma * recon_loss - discriminator_loss

        dec_grads = dec_tape.gradient(gan_loss, self.vae.decoder.trainable_variables)
        self.dec_optimizer.apply_gradients(zip(dec_grads, self.vae.decoder.trainable_variables))

        # Update loss trackers
        self.vae_loss_tracker.update_state(encoder_loss + gan_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        return {m.name: m.result() for m in self.metrics}

    def generate(self, num_samples, scale):
        z_samples = tf.random.normal([num_samples, self.latent_dim]) * scale
        synthetic_data = self.vae.decoder(z_samples)
        synthetic_data = synthetic_data.numpy()
        synthetic_data = (synthetic_data - np.mean(synthetic_data, axis=0)) / np.std(synthetic_data, axis=0)
        return synthetic_data