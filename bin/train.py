from lib.prep_data import prepare_data
from lib.parse_confs import parse_kwargs
import importlib
import os


def train_model(name):
    # This function implicitely defines a number of conventions. The first convention defined is the
    # fact that the module name of the model, the class containing the model and the config file
    # pertaining to the model should all have the same name. The next convention defined is the
    # presence of the sections "Database_Parameters" and "Model_Parameters" in all config files
    # that configure a trainable gan network. Finally the last convetion here is the relative position
    # of the bin files and the conf files that being "..\\conf\\model.cfg". One more, each model
    # should be fully defined upons instantiation and contain the function train which will be called
    # here. All trained models should be stored in a dictionary with there model_name acting as their
    # key. All train functions should take training data and epochs as inputs. Naming convention

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
    os.chdir("..\\results")
    for component_name, model in model.model.items():
        model.save(component_name)


def select_model(name):

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