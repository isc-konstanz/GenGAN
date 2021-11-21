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
    latent_target = ''.join(name.split('_')).upper()
    for model_py in os.listdir('..\\models'):
        model = os.path.splitext(model_py)[0]
        latent_name = ''.join(model.split('_')).upper()
        if latent_name == latent_target:
            module = importlib.import_module('models.'+model)
        else:
            continue

    # Retrieve the relevant model class
    try:
        letters = list(latent_target)
        class_name = letters[0] + ''.join(letters[1:-3]).lower() + ''.join(letters[-3:])
        conf_name = ''.join(letters[:-3]).lower() + '_' + ''.join(letters[-3:]).lower() + '.cfg'
        model_type = getattr(module, class_name)

    except:
        list_models = [a for a in os.listdir('..\\models') if a.endswith('gan')]
        print('Please choose one of the following implemented models:')

        i = 1
        for model in list_models:
            print('{}.'.format(i) + model)
            i += 1

    # Dynamically retrieve database configs
    base_file = os.path.join("..", "conf", "base_model.cfg")
    kwargs = parse_kwargs(base_file, "Database_Parameters")

    # prepare data and instantiate model class
    np_data = prepare_data(**kwargs)
    model_file = os.path.join("..", "conf", conf_name)
    kwargs = {**parse_kwargs(model_file, "Model_Parameters"),
              **parse_kwargs(base_file, "Model_Parameters")}
    model = model_type(**kwargs)

    model.train(np_data)

    os.chdir("..\\results")
    for component_name, model in model.model.items():
        model.save(os.path.join(class_name, component_name))

if __name__ == '__main__':
    train_model('SimpleGAN')
#main('time_gan')