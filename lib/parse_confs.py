from configparser import ConfigParser
import json


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