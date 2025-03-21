"""
Optimiser codebase: v0.1
Author: Tranter Tech
Date: 2024
"""
import sys

import session
from utils.tools import load_config, StdOutLogHandler, ErrorWrapper
from app.spooler import Spooler

import logging
import argparse

# get the arguments relevant to this app
parser = argparse.ArgumentParser("Optimiser")
parser.add_argument("--spooler",
                    action='store_true',
                    help="Determine whether this is a spooler session or not")
parser.add_argument("--learner",
                    action='store_true',
                    help="Determine whether this is learner session or not")
parser.add_argument("--sampler",
                    action='store_true',
                    help="Determine whether this learner is a sampler")
parser.add_argument("--driver_test",
                    action='store_true',
                    help="Load the drivers module and run the main test in it")
parser.add_argument("--interface_test",
                    action='store_true',
                    help="Load interface and test it")
parser.add_argument('-i', '--id', type=int,
                    help='The identifier of the learner to be used in conjunction with --learner.')
args = parser.parse_args()

# setup the logging
logging.basicConfig(level=logging.DEBUG,
                    filename='optimiser.log',
                    format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')

_LOG = logging.getLogger('Main')

# wrap the stderr so we can capture it properly
err_file = open('./stderr_spooler.txt', 'w+')
err_wrapper = ErrorWrapper(err_file, _LOG)
sys.stderr = err_wrapper

if __name__ == '__main__':
    _CONFIG = load_config(_LOG)

    if bool(args.spooler):

        session.display_log_handler = StdOutLogHandler()
        spooler = Spooler()
        spooler.initialise_spooler()

        x = input('>')

        spooler.close()

    elif bool(args.learner):
        import os
        import psutil
        pid = os.getpid()
        p = psutil.Process(os.getpid())

        from app.learner import NeuralNetLearner, SamplingLearner
        session.display_log_handler = StdOutLogHandler()

        if args.id is None:
            raise RuntimeError('Need an integer value for the id of the process.')

        if args.sampler:
            sl = SamplingLearner(f'SL{args.id}')
            sl.initialise()
        else:
            p.cpu_affinity([int(args.id)])
            nnl = NeuralNetLearner(f'NN{args.id}')
            nnl.initialise()

    elif bool(args.driver_test):
        # fix the logging
        session.display_log_handler = StdOutLogHandler()
        _LOG.addHandler(session.display_log_handler)
        _LOG.info('Specified main.py as driver test.')

        _LOG.info(f"Devices in config: {_CONFIG.get('devices', [])}")
        from app import drivers

        # run the main function
        drivers.main()

    elif bool(args.interface_test):
        session.display_log_handler = StdOutLogHandler()
        _LOG.addHandler(session.display_log_handler)
        _LOG.info('Specified main.py as interface test.')

        from app.interface import test
        test()

    # -------------------------------------
    # Shutdown
    # -------------------------------------
    # finally shutdown what we need to
    try:
        err_file.close()
    except Exception as e:
        _LOG.error(f'Failed to close stderr file: {e.args}')