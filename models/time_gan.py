"""
TimeGAN class implemented accordingly with:
Original code can be found here: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/
"""
from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor,float32
from tensorflow import data as tfdata
from tensorflow import nn
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from .base_model import BaseModel

import numpy as np
from tqdm import tqdm, trange

def make_net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type=='GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                      return_sequences=True,
                      name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model


class TimeGAN(BaseModel):
    def __init__(self, targets, batch_size, lr, beta_1, beta_2, seq_len, hidden_dim, gamma, epochs, **kwargs):

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n_seq = len(targets)
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epochs = epochs
        # ToDo: consider putting kwargs in __init__
        super().__init__(targets=targets, batch_size=batch_size, seq_len=seq_len, epochs=epochs)

        self.define_gan()
        self.model = {'generator': self.generator_aux, 'supervisor': self.supervisor,
                      'discriminator': self.discriminator, 'recovery': self.recovery,
                      'embedder': self.embedder}

    def define_gan(self):

        self.generator_aux = Generator(self.hidden_dim).build()
        self.supervisor = Supervisor(self.hidden_dim).build()
        self.discriminator = Discriminator(self.hidden_dim).build()
        self.recovery = Recovery(self.hidden_dim, self.n_seq).build()
        self.embedder = Embedder(self.hidden_dim).build()

        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        #--------------------------------
        # Building the AutoEncoder
        #--------------------------------
        e_X = self.embedder(X)
        re_X = self.recovery(e_X)

        self.autoencoder = Model(inputs=X, outputs=re_X)

        #---------------------------------
        # Adversarial Supervise Architecture
        #---------------------------------
        g_Z = self.generator_aux(Z)
        sg_Z = self.supervisor(g_Z)
        dsg_Z = self.discriminator(sg_Z)

        self.dsg_Z_model = Model(inputs=Z,
                                 outputs=dsg_Z,
                                 name='AdversarialSupervised')

        #---------------------------------
        # Adversarial architecture in latent space
        #---------------------------------
        dg_Z = self.discriminator(g_Z)

        self.dg_Z_model = Model(inputs=Z,
                                outputs=dg_Z,
                                name='AdversarialEmbedded')
        # ---------------------------------
        # Synthetic data generation
        # ---------------------------------
        rsg_Z = self.recovery(sg_Z)
        self.rsg_Z_model = Model(inputs=Z,
                                 outputs=rsg_Z,
                                 name='FinalGenerator')

        # --------------------------------
        # Final discriminator model
        # --------------------------------
        de_X = self.discriminator(e_X)
        self.de_X_model = Model(inputs=X,
                                outputs=de_X,
                                name="RealDiscriminator")

        # ----------------------------
        # Define the loss functions
        # ----------------------------
        self._mse=MeanSquaredError()
        self._bce=BinaryCrossentropy()


    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            re_x = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, re_x)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, z, opt):
        with GradientTape() as tape:
            e_x = self.embedder(x)
            g_z = self.generator_aux(z)
            sg_z = self.supervisor(g_z)
            generator_loss_supervised = self._mse(e_x, sg_z)

        # ToDo it should be considered whether the generator should be allowed to change its weights
        # so as to improve the supervised loss function, or whether the generator should solely
        # concern itself with the adversarial loss of the network.
        var_list = self.supervisor.trainable_variables + self.rsg_Z_model.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @function
    def train_embedder(self, x, z, opt):
        with GradientTape() as tape:
            # Supervised Loss
            e_x = self.embedder(x)
            g_z = self.generator_aux(z)
            sg_z = self.supervisor(g_z)
            generator_loss_supervised = self._mse(e_x, sg_z)

            # Reconstruction Loss
            re_x = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, re_x)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z):
        # Loss on false negatives
        y_real = self.de_X_model(x)
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        # Loss on false positives
        y_fake = self.dsg_Z_model(z)
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        y_fake_e = self.dg_Z_model(z)
        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @function
    def train_generator(self, x, z, opt):
        with GradientTape() as tape:
            dsg_z = self.dsg_Z_model(z)
            dsg_loss_generator = self._bce(y_true=ones_like(dsg_z),
                                                    y_pred=dsg_z)

            dg_z = self.dg_Z_model(z)
            dg_loss_generator = self._bce(y_true=ones_like(dg_z),
                                                      y_pred=dg_z)
            e_x = self.embedder(x)
            g_z = self.generator_aux(z)
            sg_z = self.supervisor(g_z)
            sg_loss = self._mse(e_x, sg_z)

            rsg_z = self.rsg_Z_model(z)
            generator_moment_loss = self.calc_generator_moments_loss(x, rsg_z)

            generator_loss = (dsg_loss_generator +
                              dg_loss_generator +
                              100 * sqrt(sg_loss) +
                              100 * generator_moment_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return dsg_loss_generator, sg_loss, generator_moment_loss

    @function
    def train_discriminator(self, x, z, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data).shuffle(buffer_size=n_windows).repeat())

    def _generate_noise(self):
        while True:
            yield np.random.normal(loc=0, scale=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size)
                                .repeat())

    def train(self, data):
        ## Embedding network training
        autoencoder_opt = Adam(learning_rate=self.lr)
        for _ in tqdm(range(self.epochs), desc='Emddeding network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)

        ## Supervised Network training
        supervisor_opt = Adam(learning_rate=self.lr)
        for _ in tqdm(range(self.epochs), desc='Supervised network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            Z_ = next(self.get_batch_noise())
            step_g_loss_s = self.train_supervisor(X_, Z_, supervisor_opt)

        ## Joint training
        generator_opt = Adam(learning_rate=self.lr)
        embedder_opt = Adam(learning_rate=self.lr)
        discriminator_opt = Adam(learning_rate=self.lr)

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for _ in tqdm(range(self.epochs), desc='Joint networks training'):

            #Train the generator (k times as often as the discriminator)
            # Here k=2
            for _ in range(2):
                X_ = next(self.get_batch_data(data, n_windows=len(data)))
                Z_ = next(self.get_batch_noise())
                # --------------------------
                # Train the generator
                # --------------------------
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, generator_opt)

                # --------------------------
                # Train the embedder
                # --------------------------
                step_e_loss_t0 = self.train_embedder(X_, Z_, embedder_opt)

            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            Z_ = next(self.get_batch_noise())
            step_d_loss = self.discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = next(self.get_batch_noise())
            records = self.rsg_Z_model(Z_)
            data.append(records)
        return np.array(np.vstack(data))


class Generator(Model):

    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self):
        model = Sequential(name='Generator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim,
                         net_type=self.net_type)
        return model

class Discriminator(Model):

    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type=net_type

    def build(self):
        model = Sequential(name='Discriminator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=1,
                         net_type=self.net_type)
        return model

class Recovery(Model):

    def __init__(self, hidden_dim, n_seq):
        self.hidden_dim=hidden_dim
        self.n_seq=n_seq
        return

    def build(self):
        recovery = Sequential(name='Recovery')
        recovery = make_net(recovery,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.n_seq)
        return recovery

class Embedder(Model):

    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim
        return

    def build(self):
        embedder = Sequential(name='Embedder')
        embedder = make_net(embedder,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.hidden_dim)
        return embedder

class Supervisor(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim

    def build(self):
        model = Sequential(name='Supervisor')
        model = make_net(model,
                         n_layers=2,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim)
        return model
