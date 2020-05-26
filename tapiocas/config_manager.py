import os
import json
import logging

from constants import DEFAULT_CONFIG_FILE, CUSTOM_CONFIG_FILE, DEFAULT_CONFIG_FOLDER


class Configuration:
    # make it a singleton
    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
        return cls._instance

    phone_ip = None
    adbkey_path = None
    output_dir = None
    log_dir = None
    log_level = None

    def load_json(self, config_file_path, override_only=False):
        if not os.path.exists(config_file_path):
            logging.warning(f"{config_file_path} does not exist")
            return

        with open(config_file_path, 'r') as f:
            contents = json.load(f)
            for key in contents:
                if override_only and key not in self.__dict__:
                    logging.warning(f"Ignoring '{key}' because it is not an expected configuration item")
                else:
                    self.__dict__[key] = contents[key]
        self._loaded = True


def get_configuration(custom_file=CUSTOM_CONFIG_FILE, config_folder=None):
    config = Configuration()
    if config_folder is None:
        config_folder = DEFAULT_CONFIG_FOLDER
    if not config._loaded:
        default_config_path = os.path.join(config_folder, DEFAULT_CONFIG_FILE)
        logging.info(f"Loading default configuration: {default_config_path}")
        config.load_json(default_config_path)
        if custom_file:
            custom_file = os.path.join(config_folder, custom_file)
            logging.info(f"Loading custom configuration: {custom_file}")
            config.load_json(custom_file, override_only=True)
    if not config.phone_ip:
        raise ValueError("Phone ip should be defined")
    return config


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    configuration = get_configuration()
    print(configuration.__dict__)

