import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv1DTranspose, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from .base_model import BaseModel

import numpy as np
import time



class SimpleGAN(BaseModel):

    def __init__(self, targets, batch_size, seq_len, n_seq, seed_size, epochs, critic_steps, clip):

        self.seed_size = seed_size
        self.n_seq = n_seq
        self.seq_len= seq_len
        self.output_dim = (seq_len, n_seq)
        self.critic_steps = critic_steps
        self.generator_optimizer = RMSprop(learning_rate=0.00005)
        self.critic_optimizer = RMSprop(learning_rate=0.00005)
        self.c = clip

        super().__init__(targets, seq_len, batch_size, epochs)

        # Model
        self.generator = None
        self.critic = None
        self.model = None

        self.build_model()

    def build_model(self):

        self._build_generator()
        self._build_critic()
        self.model = {'generator': self.generator, 'critic': self.critic}

    def _build_generator(self):

        model = Sequential()

        #model.add(Conv1DTranspose(512, kernel_size=3, input_shape=(6, 1)))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))

        model.add(Conv1DTranspose(256,
                                  kernel_size=3,
                                  input_shape=(self.seed_size, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv1DTranspose(128,
                                  kernel_size=3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv1DTranspose(64,
                                  kernel_size=3,
                                  strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv1DTranspose(32,
                                  kernel_size=3,
                                  strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv1DTranspose(1,
                                  kernel_size=3,
                                  strides=2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        self.generator = model

    def _build_critic(self):

        model = Sequential()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.c/2)

        model.add(Conv1D(32,
                         kernel_size=3,
                         strides=2,
                         input_shape=self.output_dim,
                         kernel_initializer=initializer))
        model.add(LeakyReLU(alpha=0.2))

        # Dropout is only required here, for the descriminator, because it is the only part
        # of the network attempting to fit a function.
        model.add(Dropout(0.25))
        model.add(Conv1D(64,
                         kernel_size=3,
                         strides=2,
                         kernel_initializer=initializer))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv1D(128,
                         kernel_size=3,
                         strides=2,
                         kernel_initializer=initializer))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1,
                        activation=None,
                        kernel_initializer=initializer))

        self.critic = model

    def critic_loss(self, real_output, fake_output):

        real_loss = tf.math.reduce_mean(real_output)
        fake_loss = tf.math.reduce_mean(fake_output)
        total_loss = - (real_loss - fake_loss)

        return total_loss

    def generator_loss(self, fake_output):

        return -tf.math.reduce_mean(fake_output)

    @tf.function
    def train_step_critic(self, forecasts):

        # forecasts consists of a batch of forecasts. Now: (32, 96, 11)
        seed = tf.random.normal([self.batch_size, self.seed_size, 1])

        with tf.GradientTape() as critic_tape:

            generated_forecasts = self.generator(seed, training=True)

            real_output = self.critic(forecasts, training=True)
            fake_output = self.critic(generated_forecasts[:, :self.output_dim[0], :], training=True)

            loss = self.critic_loss(real_output, fake_output)

            gradient = critic_tape.gradient(loss, self.critic.trainable_variables)

            self.critic_optimizer.apply_gradients(zip(gradient,
                                                      self.critic.trainable_variables))

            for layer_weights in self.critic.trainable_variables:
                if not layer_weights.name.startswith('batch'):
                    layer_weights.assign(tf.clip_by_value(layer_weights, -self.c, self.c))

        return loss

    @tf.function
    def train_step_generator(self):

        # forecasts consists of a batch of forecasts. Now: (32, 96, 11)
        seed = tf.random.normal([self.batch_size, self.seed_size, 1])

        with tf.GradientTape() as gen_tape:

            forecasts = self.generator(seed, training=True)
            loss = self.generator_loss(forecasts[:, :self.output_dim[0], :])
            gradient = gen_tape.gradient(loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradient,
                                                         self.generator.trainable_variables))
        return loss

    def train(self, np_data):

        start = time.time()

        for epoch in range(self.epochs):

            gen_loss_list = []
            critic_loss_list = []

            train_data = np_data
            np.random.shuffle(train_data)

            for j in range(len(train_data)):

                critic_loss = self.train_step_critic(train_data[j])
                critic_loss_list.append(critic_loss)

                if j % 5 == 0 and j != 0:
                    gen_loss = self.train_step_generator()
                    gen_loss_list.append(gen_loss)

            gen_loss = sum(gen_loss_list) / len(gen_loss_list)
            critic_loss = sum(critic_loss_list) / len(critic_loss_list)

            print(f'Epoch {epoch + 1}, gen loss = {gen_loss}, crit loss = {critic_loss}.')

        elapsed = time.time() - start
        print(f'Training time: '.format(elapsed))

if __name__ == '__main__':

    model = SimpleGAN(targets=["el_power"],
                      batch_size=128, seq_len=24,
                      n_seq=1, epochs=10,
                      seed_size=24, critic_steps=5,
                      clip=0.01)