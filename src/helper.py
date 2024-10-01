import time
from datetime import datetime, timezone, timedelta
import os
import math
import numpy as np
from gridappsd.field_interface.interfaces import MessageBusDefinition
import json
import toml

import logging
import sys

log = logging.getLogger(__name__)

#### Log configuration
def configure_logging(level: str, file_name: str, format: str, datefmt: str):
    """Configures the python logging

    Besides configuring the python logging this functions also configure untracked exceptions to also be written to the log file.

    Args:
        level (str): logging level
        file_name (str): path and name of file to have the log
        format (str): format to write to log file
        datefmt (str): date and time format for the log file
    """    
    level = getattr(logging, level.upper(), logging.DEBUG)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear any default handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Create a file handler and set level
    file_handler = logging.FileHandler(file_name, mode='w')
    file_handler.setLevel(level)
    
    # Set the formatter for the file handler
    formatter = logging.Formatter(format, datefmt)
    file_handler.setFormatter(formatter)
    
    # Add file handler to the root logger
    root_logger.addHandler(file_handler)
    
    # Create a console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    
    # Add console handler to the root logger
    root_logger.addHandler(console_handler)
    
    # Set the root logger level
    root_logger.setLevel(level)
    
    # Set the global exception handler
    def log_exception(exc_type, exc_value, exc_traceback):
        root_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = log_exception

#### Time
def human_time(time: str) -> datetime:
    """Convert string number to date and time

    The conversions also includes the time zone. The timezone is hard codded to Coordinated Universal Time (UTC).

    Args:
        time (str): string number representation of unix time

    Returns:
        datetime: date and time representation of the unix number date and time
    """    
    human_time = datetime.fromtimestamp(float(time),tz=timezone.utc)
    return human_time

def comp_time(time: datetime) -> str:
    """Converts date and time to string number representation unix time

    Args:
        time (datetime): date and time to be converted to unix time

    Returns:
        str: unix time of the date and time
    """    
    timestamp = str(time.timestamp())
    return timestamp

#### Input files
def load_json(path: str) -> dict:
    """Reads text file of type json

    Args:
        path (str): path and file name to be read

    Returns:
        dict: dictionary representation of the data read
    """    
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)

def replace_json_references(d: dict): 
    """Update dictionary with json file

    For keys ending in .json the path and file name specified are read from the respective file and replaced in the dictionary.

    Args:
        d (dict): dictionary with possible values strings ending in .json
    """    
    for key, value in d.items():
        if isinstance(value, dict):
            # Recurse into nested dictionaries
            replace_json_references(value)
        elif isinstance(value, str) and value.endswith('.json'):
            d[key] = load_json(value)

def substitute_core_variables(config:dict, core:dict) -> dict:
    """Update dictionary value strings with f-string type formatting

    The information to be replaced in the config dictionary is part of the core dictionary. Not using other variables other than the core dictionary for the updates.

    Args:
        config (dict): dictionary with f-string type string to be updated
        core (dict): dictionary with the data to be replaced in the f-string

    Returns:
        dict: updated config dictionary
    """    
    # Create a new dictionary to hold the substituted values
    substituted_config = {}
    # Iterate over the key-value pairs in the config
    for key, value in config.items():
        # print(key)
        # print(value)
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            substituted_config[key] = substitute_core_variables(value, core)
        elif isinstance(value, str):
            # Replace placeholders in strings with core variable values
            substituted_config[key] = value.format(**core)
        else:
            # For non-string values, keep them as they are
            substituted_config[key] = value
    return substituted_config

def load_toml(path: str,core_key: str='core') -> dict:
    """Reads text file of type toml

    The toml file is expected to have a key of `core` values to be replaced in a f-string stile for some of the other entries. This functions:
        - reads the toml file 
        - calls for the update of the f-string
        - reads the requested json files

    Args:
        path (str): path and name of file to be read
        core_key (str, optional): key of read dictionary to be utilized as core variables for f-string type update. Defaults to 'core'.

    Returns:
        dict: contains the data in toml and other files requested in the toml file
    """    
    with open(path, 'r') as f:
        dictionary = toml.load(f)
    #### Updating with core
    if core_key in dictionary:
        core=dictionary[core_key]
        del dictionary[core_key]
        for key, value in core.items():
            core[key] = os.path.expanduser(value)
        dictionary = substitute_core_variables(dictionary, core)
    #### adding the json file information
    replace_json_references(dictionary)
    return dictionary

def os_variables(dictionary: dict) -> None:
    """Populate environmental variables

    Args:
        dictionary (dict): key is the name of the environmental variable to be created and the value is the data to be contained in the variable.
    """    
    for key, value in dictionary.items():
        os.environ[key] = value

def pol_to_cart(rho: float, phi: float) -> complex:
    """Convert polar complex value format of two  floats to a single complex number

    Args:
        rho (float): complex magnitude
        phi (float): complex polar angle in degrees

    Returns:
        complex: complex number
    """    
    rad = math.radians(phi)
    x = rho * np.cos(rad)
    y = rho * np.sin(rad)
    return complex(x, y)

def overwrite_parameters(feeder_id: str, area_id: str = "") -> MessageBusDefinition:
    """Overwrite parameters for message bus definition GridAPPSD

    Args:
        feeder_id (str): GridAPPSD feeder id
        area_id (str, optional): Area of the GridAPPSD message bus. Defaults to "".

    Raises:
        ValueError: missing initialization of variables to be able to utilize this function

    Returns:
        MessageBusDefinition: created GridAPPSD message bus for the specified feeder id and area
    """    
    bus = MessageBusDefinition.load(os.environ.get("BUS_CONFIG"))
    if area_id:
        bus.id = feeder_id + "." + area_id
    else:
        bus.id = feeder_id

    address = os.environ.get("GRIDAPPSD_ADDRESS")
    port = os.environ.get("GRIDAPPSD_PORT")
    if not address or not port:
        raise ValueError("import auth_context or set environment up before this statement.")
    bus.conneciton_args["GRIDAPPSD_ADDRESS"] = f"tcp://{address}:{port}"
    bus.conneciton_args["GRIDAPPSD_USER"] = os.environ.get("GRIDAPPSD_USER")
    bus.conneciton_args["GRIDAPPSD_PASSWORD"] = os.environ.get("GRIDAPPSD_PASSWORD")
    return bus
       
if __name__ == "__main__":
    #### substitute_core_variables
    config = {
        "core": {
            "base_path": "~/grid/dltdflisr",
            "another_var": "some_value"
        },
        "settings": {
            "option1": True,
            "option2": 42
        },
        "paths": {
            "file": "{base_path}/config/message_config_0.json",
            "another_path": "{base_path}/data/{another_var}/file.txt",
            "static_path": "/static/assets",
            "nested": {
                "deep_file": "{base_path}/deep/nested/config_{another_var}.json"
            }
        }
    }
    core=config['core']
    del config['core']
    for key, value in core.items():
        core[key] = os.path.expanduser(value)
    new_config=substitute_core_variables(config,core)
    print(new_config)
    #### adding json file configuration
    file_location = os.path.dirname(os.path.abspath(__file__))
    config = {'file':f'{file_location}/../config/message_config_0.json'}
    replace_json_references(config)
    print(config)
    #### time conversion
    temp="1563872400.00005"
    print(temp)
    temp=human_time(temp)
    print(temp)
    temp=comp_time(temp)
    print(temp)
