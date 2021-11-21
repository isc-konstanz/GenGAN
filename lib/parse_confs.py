from configparser import ConfigParser
import json

def parse_kwargs(file, section):
    parameters = ConfigParser()
    parameters.read(file)

    kwargs = dict(parameters[section].items())

    for label, item in kwargs.items():
        kwargs[label] = json.loads(item)

    return kwargs