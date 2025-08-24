import configparser
from pathlib import Path

def load_config():
    config = configparser.ConfigParser()
    ini_path = Path(__file__).with_name("uiconfigfile.ini")
    if ini_path.exists():
        config.read(ini_path)
    return config
