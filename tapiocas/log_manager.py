import logging
import logging.config
import os
import shutil
import datetime


def initialize_log(log_folder, log_file_name='tapiocas', log_level='DEBUG'):
    os.makedirs(log_folder, exist_ok=True)
    file_name = os.path.join(log_folder, f'{log_file_name}.log')
    if os.path.exists(file_name):
        archive_folder = os.path.join(log_folder, 'archive')
        archive_file(file_name, archive_folder, log_file_name)
    set_logging_config(file_name, log_level)


def archive_file(file_path, archive_folder, log_file_name):
    os.makedirs(archive_folder, exist_ok=True)
    suffix = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(file_path, os.path.join(archive_folder, f'{log_file_name}_{suffix}.log'))
    os.remove(file_path)


def set_logging_config(file_name, log_level):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s.%(msecs)03d [%(levelname)s] %(filename)s-%(funcName)s : %(message)s',
                'datefmt': '%H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': log_level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': log_level,
                'filename': file_name,
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
            },

        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': True
            }
        }
    })
