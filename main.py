"""
Optimiser codebase: v0.1
Author: Tranter Tech
Date: 2024
"""
import session
from utils.Tools import DisplayLogHandler, load_config, StdOutLogHandler
from app.display import MainDisplay
from app.manager import Manager
from app.interface import TestInterface
from comms import TCP
from app.spooler import Spooler

import logging
import argparse
from threading import Thread


# declare the main display
md = MainDisplay()
session.display_log_handler = DisplayLogHandler(md.external_log)

# get the arguments relevant to this app
parser = argparse.ArgumentParser("Optimiser")
parser.add_argument("--client",
                    action='store_true',
                    help="Determine whether this is a client session or not")
parser.add_argument("--learner",
                    action='store_true',
                    help="Determine whether this is learner session or not")
parser.add_argument("--sampler",
                    action='store_true',
                    help="Determine whether this learner is a sampler")
parser.add_argument("--driver_test",
                    action='store_true',
                    help="Load the drivers module and run the main test in it")
parser.add_argument('-i', '--id', type=int,
                    help='The identifier of the learner to be used in conjunction with --learner.')
args = parser.parse_args()

# setup the logging
logging.basicConfig(level=logging.DEBUG,
                    filename='optimiser.log',
                    format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')
_LOG = logging.getLogger('Main')
_LOG.addHandler(session.display_log_handler)

if __name__ == '__main__':
    _CONFIG = load_config(_LOG)

    if bool(args.client):

        session.display_log_handler = StdOutLogHandler()
        spooler = Spooler()
        spooler.initialise_spooler()

        x = input('>')

        spooler.close()

    elif bool(args.learner):
        import os
        print(os.getpid())
        from app.learner import NeuralNetLearner, SamplingLearner
        session.display_log_handler = StdOutLogHandler()

        if args.id is None:
            raise RuntimeError('Need an integer value for the id of the process.')

        if args.sampler:
            sl = SamplingLearner(f'SL{args.id}')
            sl.initialise()
        else:
            nnl = NeuralNetLearner(f'NN{args.id}')
            nnl.initialise()

    elif bool(args.driver_test):
        # fix the logging
        session.display_log_handler = StdOutLogHandler()
        _LOG.addHandler(session.display_log_handler)
        _LOG.info('Specified main.py as driver test.')

        _LOG.info(f"Devices in config: {_CONFIG.get('devices', [])}")
        from app import drivers

    else:
        # start the display thread
        display_thread = Thread(target=md.start_display)
        display_thread.start()

        # declare the TCP server, the manager will start it
        server = TCP.Server(_CONFIG)

        # declare the manager to run the optimisation
        manager = Manager(server)
        optimisation_config = {'bound_restriction': '0.05',
                               'initial_count': '3',
                               'learner_number': '2',
                               'halt_number': '100',
                               'bounds': tuple([(3, 12)]*5),
                               'interface': TestInterface(),
                               'interface_args': 'ackley'
                               }
        manager.initialise_optimisation(optimisation_config)
        manager.start_optimisation()

        # when we are finished, join the display thread
        display_thread.join()

        # ditch the display and write to stdout
        manager.log.addHandler(StdOutLogHandler())

        # exit steps
        manager.close()
