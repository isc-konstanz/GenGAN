from configparser import ConfigParser
import json
import os


def parse_kwargs(file, section):
    """loads selected sections of a config file

    :param file: path to config file
    :type file: str

    :param section: name of desired section
    :type section: str

    :return kwargs: parameters contained in section
    :rtype kwargs: dict
    """

    parameters = ConfigParser()
    parameters.read(file)

    kwargs = dict(parameters[section].items())

    for label, item in kwargs.items():
        kwargs[label] = json.loads(item)

    return kwargs


def select_model(name):
    """returns conf, module, and class names for a selected model

    :param name: string indicating the selected model
    :type name: str

    :return kwargs: dictionary containing the conf, module, and
    class names associated with the selected model
    :rtype kwargs: dict

    ..notes:: This function explicitely defines the convention that
    all models are assumed to end in 'gan' (not case sensitive) and
    be preceded by one word (e.g. TimeGAN, time_gan, time_GAN).
    """

    latent_target = ''.join(name.split('_')).upper()
    models = []
    for model_file in os.listdir('..\\conf'):

        model_name = os.path.splitext(model_file)[0]
        models.append(model_name)
        latent_name = ''.join(model_name.split('_')).upper()
        if latent_name == latent_target:
            model_file = os.path.join('..\\conf', model_file)
            return parse_kwargs(file=model_file, section='Aliases')
        else:
            continue

    raise NotImplementedError('Please choose one of the following implemented'
                                'models:\n' + '\n'.join(models))