"""

Author: Tranter Tech
Date: 2024
"""
import json
import logging
from datetime import datetime
from enum import Enum

def load_config(log):
    """
    Load the config file and throw an error if not possible
    :param log: logger to report to.
    :return: (dict) config file
    """
    try:
        with open('config.json', 'r') as fp:
            config = json.load(fp)
            return config
    except FileNotFoundError:
        log.error('Config file not found. Check the existence of ../config.json.')
    except json.decoder.JSONDecodeError:
        log.error('Couldn\'t decode the JSON config. Check file integrity.')
    except Exception as e:
        log.error(f'Unknown exception occurred:')
        for err in e.args:
            log.error(f'\t{err}')
            raise RuntimeError('Unhandled exception loading the config file.')


class DisplayLogHandler(logging.Handler):
    def __init__(self, md_func):
        super().__init__()

        # display function that connects to the terminal
        self.display_func = md_func

    def emit(self, record):
        # update the text colour
        text_colour = 1
        if record.levelname == 'ERROR':
            text_colour = 3
        elif record.levelname == 'WARNING':
            text_colour = 2

        # try to override the colour if we can
        try:
            text_colour = record.colour
        except AttributeError as e:
            pass

        # handle the display of text to the terminal
        self.display_func(f'[{record.levelname}][{datetime.fromtimestamp(record.created).ctime()}]'
                          f'[{record.name}] {record.msg}', colour=text_colour)


class StdOutLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()

        # display function that connects to the terminal
        self.display_func = print

    def emit(self, record):
        # update the text colour
        text_colour = ''
        if record.levelname == 'ERROR':
            text_colour = '\033[93m'
        elif record.levelname == 'WARNING':
            text_colour = '\033[91m'

        # handle the display of text to the terminal
        self.display_func(f'{text_colour}'
                          f'[{record.levelname}][{datetime.fromtimestamp(record.created).ctime()}]'
                          f'[{record.name}] {record.msg}'
                          f'\033[0m')


class LearnerState(int, Enum):
    PRE_INIT = 0
    INIT = 1
    CONFIGURED = 2
    READY = 3
    TRAIN = 4
    MINIMISING = 5
    PREDICT = 6
    PREDICTED = 7
    BUSY = 8


class ErrorWrapper:
    def __init__(self, base_obj, log):
        self.base_obj = base_obj
        self.log = log

    # keep writing the std_error to the text file while managing the display
    def check_write(self, text):
        # filter the error string
        filtered = text.strip('\n\t ')
        if len(filtered) > 0:
            self.log.error(filtered)
        return self.base_obj.write(text)

    # get the attribute and wrap the write function
    def __getattr__(self, name):
        if name == 'write':
            return self.check_write
        else:
            return getattr(self.base_obj, name)
