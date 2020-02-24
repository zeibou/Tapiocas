import logging
import logging.config
import os
import shutil
import datetime

def initialize_log(log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    file_name = os.path.join(log_folder, f'adb_connector.log')
    if os.path.exists(file_name):
        archive_folder = os.path.join(log_folder, 'archive')
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        shutil.copyfile(file_name, os.path.join(archive_folder,
                                                'adb_connector_{0}.log'.format(datetime.datetime.fromtimestamp(
                                                    os.path.getctime(file_name)).strftime("%Y%m%d_%H%M%S"))))

        os.remove(file_name)
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
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'filename': file_name,
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
            },

        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })
