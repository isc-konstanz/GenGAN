from lib.prep_data import prepare_data
from lib.parse_confs import parse_kwargs, select_model
import importlib
import os


def retrieve_params(name):
    """retrieve parameters required for model instantiation, training, and evaluation

    :param name: name of desired GAN model
    :type name: str

    :return aliases: dictionary containing the conf, module, and
    class names associated with the selected model.
    :rtype aliases: dict

    :return data_params: dictionary specifying how the database should be parsed
    :rtype aliases: dict

    :return nn_params: dictionary specifying network paramaters required for instantiation
    :rtype aliases: dict

    :return eval_params: dictionary specifying which evaluation functions present in the
    evaluate module should be carried out
    :rtype eval_params: dict

    ..notes:: The sections 'Aliases', 'Model_Parameters' and 'Evaluation' must
    be present in all config files, the singular exception being the base_model.cfg. This
    config must contain simply the sections 'Database_Parameters' and 'Model_Parameters'.
    """
    aliases = select_model(name)

    # Dynamically retrieve database configs
    base_file = os.path.join("..", "conf", "base_model.cfg")
    data_params = parse_kwargs(base_file, "Database_Parameters")

    # prepare data and instantiate model class
    model_file = os.path.join("..", "conf", aliases['conf_name'])
    nn_params = {**parse_kwargs(model_file, "Model_Parameters"),
                 **parse_kwargs(base_file, "Model_Parameters")}

    eval_params = parse_kwargs(model_file, "Evaluation")

    return aliases, data_params, nn_params, eval_params


def instantiate_model(aliases, nn_params):
    """instantiates a GAN model for data generation

    :return aliases: dictionary containing the conf, module, and
    class names associated with the selected model.
    :rtype aliases: dict

    :return nn_params: dictionary specifying network paramaters required for instantiation
    :rtype aliases: dict
    """
    # Dynamically retrieve module corresponding to name
    module = importlib.import_module(('models.'+aliases['module_name'].split('.')[0]))
    model_type = getattr(module, aliases['class_name'])

    model = model_type(**nn_params)

    return model


def save_model(model):
    """saves model to the results directory

    :param model: dictionary containing keras.model objects
    :type model: dict

    ..notes:: The function implicitely defines the convention that
    all model classes should store the objects describing their
    components in a dictionary called model. Futhermore, it saves the
    trained models components to the results directory; if any components
    saved have the same name as the components of a previous model trained
    by this function, the previous results will be overwritten.
    """
    os.chdir("..\\results")
    for component_name, component in model.items():
        component.save(component_name)


def main(model_name):

    model_files, data_params, nn_params, eval_params = retrieve_params(model_name)
    test_gan = instantiate_model(model_files, nn_params)
    np_data = prepare_data(**data_params)
    test_gan.train(np_data)
    save_model(test_gan.model)


if __name__ == '__main__':

    main('SimpleGan')
