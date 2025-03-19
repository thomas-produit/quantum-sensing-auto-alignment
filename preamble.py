"""
Preamble for starting an optimisation. Mostly just for keeping things clean
Author: Tranter Tech
Date: 2024
"""
import sys
import session
from utils.tools import DisplayLogHandler, load_config, StdOutLogHandler, ErrorWrapper
from app.display import MainDisplay
from app.manager import Manager
from comms import TCP

import logging
from threading import Thread


def execute():
    # declare the main display
    md = MainDisplay()
    session.display_log_handler = DisplayLogHandler(md.external_log)

    # setup the logging
    logging.basicConfig(level=logging.DEBUG,
                        filename='optimiser.log',
                        format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')

    log = logging.getLogger('Main')

    # wrap the stderr so we can capture it properly
    err_file = open('./stderr.txt', 'w+')
    err_wrapper = ErrorWrapper(err_file, log)
    sys.stderr = err_wrapper

    # pull in the config and log any details
    config = load_config(log)

    # start the display thread
    display_thread = Thread(target=md.start_display)
    display_thread.start()

    # declare the TCP server, the manager will start it
    server = TCP.Server(config)

    # declare the manager to run the optimisation
    manager = Manager(server, runtime_config=config, run_graphical=False)

    # store the required variables
    session.manager = manager
    session.display_thread = display_thread
