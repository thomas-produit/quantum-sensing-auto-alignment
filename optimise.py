"""
Main entry point for the optimisation client. This imports the minimum libraries and begins the optimisation
Author: Tranter Tech
Date: 2024
"""
import session
import preamble
from utils.tools import StdOutLogHandler
from app.interface import TestInterface

# load everything from the preamble
preamble.execute()
manager = session.manager
display_thread = session.display_thread

# optimisation config
optimisation_config = {'bound_restriction': '0.01',
                       'initial_count': '100',
                       'learner_number': '1',
                       'halt_number': '200',
                       'bounds': tuple([(-32, 32)]*5),
                       'interface': TestInterface(),
                       'interface_args': 'ackley',
                       'learner_min_tol': 1E-5
                       }

# initialise the optimisation and start it
manager.initialise_optimisation(optimisation_config)
manager.start_optimisation()

# when we are finished, join the display thread
display_thread.join()

# ditch the display and write to stdout
manager.log.addHandler(StdOutLogHandler())

# exit steps
manager.close()
