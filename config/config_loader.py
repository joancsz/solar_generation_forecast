import yaml
import os

config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config():
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
