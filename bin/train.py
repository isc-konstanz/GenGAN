from lib.prep_data import prepare_data
from lib.parse_confs import parse_kwargs
import importlib
import os


def train_model(name):
    """trains a GAN model for data generation

    :param name: name of desired GAN model
    :type name: str

    ..notes:: This function saves the trained models components to the results
    directory if any components saved have the same name as the components of
    a previous model trained by this function, the previous results will be
    overwritten.

    This function implicitely defines a number of important conventions:

        1. The sections 'Aliases', 'Database Parameters',  and
           'Model_Parameters' must be present in all config files.

        2. The relative path between the bin dir and conf dir must be
           "..\\conf\\model.cfg".

        3. All models present in the model dir should be built upon
           instantiation, and should naturally have their own train function
           called here. Furthermore the input of the class defined train
           function should receive the batched training data and number of
           epochs as input.

        4. All model classes should store the objects describing their
           components in a dictionary called model.

        5. All models are assumed to end in 'gan' (not case sensitive) and
           be preceded by one word (e.g. TimeGAN, time_gan, time_GAN).
    """
    # Dynamically retrieve module corresponding to name
    model_aliases = select_model(name)
    module = importlib.import_module(('models.'+model_aliases['module_name'].split('.')[0]))
    model_type = getattr(module, model_aliases['class_name'])

    # Dynamically retrieve database configs
    base_file = os.path.join("..", "conf", "base_model.cfg")
    kwargs = parse_kwargs(base_file, "Database_Parameters")

    # prepare data and instantiate model class
    np_data = prepare_data(**kwargs)
    model_file = os.path.join("..", "conf", model_aliases['conf_name'])
    kwargs = {**parse_kwargs(model_file, "Model_Parameters"),
              **parse_kwargs(base_file, "Model_Parameters")}
    model = model_type(**kwargs)

    model.train(np_data)

    return model


def save_model(model):
    """saves model to the results directory

    :param model: dictionary containing keras.model objects
    :type model: dict

    ..notes:: The function implicitely defines the convention that
    all model classes should store the objects describing their
    components in a dictionary called model.
    """
    os.chdir("..\\results")
    for component_name, model in model.model.items():
        model.save(component_name)


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


if __name__ == '__main__':

    test_gan = train_model('SiMpleGAN')
    save_model(test_gan)

#main('time_gan')