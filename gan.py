import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Conv2DTranspose, LeakyReLU, GlobalMaxPooling2D, Reshape
import numpy as np
import os
import sys
import logging
tf.get_logger().setLevel(logging.FATAL)

latent_dim = 128
batch_size = 700
epochs = 100
sample_vector = tf.random.normal(shape=(1, latent_dim))

"""28x28"""
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_all = np.concatenate([x_train, x_test])
x_all = x_all.astype('float32') / 255.0
x_all = np.reshape(x_all, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(x_all)
dataset = dataset.shuffle(1024).batch(batch_size)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.layer_2 = LeakyReLU(alpha=0.2)
        self.layer_3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')
        self.layer_4 = LeakyReLU(alpha=0.2)
        self.layer_5 = GlobalMaxPooling2D()
        self.layer_6 = Dense(1)

    def call(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer_1 = Dense(7 * 7 * latent_dim)
        self.layer_2 = LeakyReLU(alpha=0.2)
        self.layer_3 = Reshape((7, 7, 128))
        self.layer_4 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')
        self.layer_5 = LeakyReLU(alpha=0.2)
        self.layer_6 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')
        self.layer_7 = LeakyReLU(alpha=0.2)
        self.layer_8 = Conv2D(1, (7, 7), padding='same', activation='sigmoid')

    def call(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        return x

discrim = Discriminator()
gen = Generator()
discrim.build((None, 28, 28, 1))
gen.build((None, latent_dim))

if (len(sys.argv) == 3):
    discrim.load_weights(sys.argv[1])
    gen.load_weights(sys.argv[2])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
d_optimizer = tf.keras.optimizers.Adam(0.0003)
g_optimizer = tf.keras.optimizers.Adam(0.0004)

d_acc = tf.keras.metrics.BinaryAccuracy()

@tf.function
def train_step (x_real):
    rand_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    x_fake = gen(rand_vectors)
    x = tf.concat([x_real, x_fake], axis=0)
    y = tf.concat([tf.zeros((x_real.shape[0], 1)), tf.ones((batch_size, 1))], axis=0)
    y_noisy = y + 0.05 * tf.random.uniform(y.shape)

    with tf.GradientTape() as tape:
        y_pred = discrim(x)
        d_loss = loss(y_noisy, y_pred)
        d_acc.update_state(y, y_pred)
    grads = tape.gradient(d_loss, discrim.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discrim.trainable_weights))

    rand_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    y_fake = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
        y_pred = discrim(gen(rand_vectors))
        g_loss = loss(y_fake, y_pred)
    grads = tape.gradient(g_loss, gen.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, gen.trainable_weights))
    return d_loss, g_loss, gen(sample_vector)[0]

for epoch in range(epochs):
    print('Start', epoch)

    for step, x_real in enumerate(dataset):
        d_loss, g_loss, imgs = train_step(x_real)
        print('Step', step)

        if step % 10 == 0:
            print('Discriminator loss:', float(d_loss))
            print('Discriminator accuracy:', float(d_acc.result()))
            print('Generator loss:', float(g_loss))

            img = tf.keras.preprocessing.image.array_to_img(imgs * 255.0, scale=False)
            img.save(os.path.join('./', str(epoch) + 'gen_img' + str(step) + '.png'))

            d_acc.reset_states()

    discrim.save_weights('discriminator_weights' + str(epoch) + '.h5')
    gen.save_weights('generator_weights' + str(epoch) + '.h5')

discrim.save('mnist_gan.h5')
