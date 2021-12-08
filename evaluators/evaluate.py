"""number of functions to be used to evaluate the trained generator of a gan network.

.. notes:: All output of each function in this module is saved to the evaluation folder.
"""
import os
import sys

import matplotlib.pyplot as plt
from tensorflow import random
import pandas as pd
import numpy as np

os.chdir('C:\\Users\\jmw\\Workspace\\th-e-gan\\evaluation')

def _create_eval_dir(name):

    if not os.path.isdir(name):
        os.mkdir(name)

    return name

def evaluate_model(model, **eval_kwargs):
    """carry out the evaluation of a model instance as configured in the configs.

    :param model: An instance of a model class defined in the models dir.
    :type model: class

    :param eval_kwargs: dictionary specifying which evaluation functions to carry
    out
    :type eval_kwargs: dict

    ..notes:: It is necessary that a class instance is passed to this function
    and not the raw tensorflow object. The reason being, that it is assumed that
    all class attributes are still accessible in each of the evaluation functions.
    """
    os.chdir('..\\evaluation')
    nn_param = model.__dict__
    batch_sample = sample(**nn_param)

    for e, value in eval_kwargs.items():
        if value == True:
            eval_func = getattr(sys.modules[__name__], e)
            eval_func(batch_sample, **nn_param)

# ToDo: All plots should plot the unscaled features.
def _plot_features(data:pd.Series, y_label:str, file:str, **nn_param):
    """plots one dimensional time series data for a given day.

    :param data: one dimensional time series data
    :type data:  pd.Series

    :param y_label: label of data amplitude
    :type y_label:  str

    :param file: string indicating where plot should be saved relative
    to the evaluation directory.
    :type file: str
    """

    plt.title(data.name)
    plt.plot(data.index, data)
    plt.ylabel(y_label)

    plt.savefig(file)
    plt.close()

def plot_batch(batch_data, **nn_param):
    from numpy.random import randint

    eval_dir = _create_eval_dir('batch_plot')
    r_list = []
    for i in range(20):
        r_list.append(randint(0, nn_param['batch_size']))

    for r_index in r_list:

        data = pd.DataFrame(batch_data[r_index], columns=nn_param['targets'])
        for target in data.columns:
            _create_eval_dir(os.path.join(eval_dir, target))
            f_name = os.path.join(eval_dir, target, 'sample {}'.format(r_index))
            _plot_features(data[target], target, f_name, **nn_param)




def sample(generator, batch_size, seq_len, n_seq, **nn_param):
    """sample the generators probability distribution over the feature space.

    :param generator: generator network of a gan network
    :type keras.Model, keras.Sequential

    :param batch_size: number of sequences per batch
    :type batch_size: int

    :param seq_len: number of timesteps per sequence
    :type seq_len: int

    :param n_seq: number of features per timestep
    :type n_seq: int

    :param targets: features to be generated by the generator
    :type targets: list

    :return sample_batch: batch of generated target features
    :rtype pd.DataFrame
    """

    noise = random.normal([batch_size, seq_len, n_seq])
    sample_batch = generator(noise, training=False)

    return sample_batch


if __name__ == '__main__':

    from bin.train import retrieve_params, instantiate_model
    from tensorflow.keras.models import load_model

    aliases, _, nn_params, eval_params = retrieve_params('TimeGan')
    model = instantiate_model(aliases, nn_params)
    model_path = os.path.join('..', 'results')
    model.generator = load_model(os.path.join(model_path, 'generator'))
    evaluate_model(model, **eval_params)

