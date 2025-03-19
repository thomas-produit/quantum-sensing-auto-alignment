"""
This file is to hold the relevant commands that are used by the program. Each command should be designated as a class
which inherits from the base command class.
"""
from enum import Enum
from datetime import datetime
import numpy as np
from time import sleep


class state(Enum):
    """
    Enum for determining indicating whether a command has run or not.
    """
    INITIAL = 0
    RUNNING = 1
    FINISHED = 2

# Not presently Used
# class LogEvent(Enum):
#     ENTERING = 0
#     LEAVING = 1


class BaseCommand:
    def __init__(self, session=None):
        """
        Base command to be inherited by the other commands. The run command should be overriden by the inheriting class
        to perform the desired action.
        :param session: Session variable which is passed by the display and stored by the command class.
        """
        self._log_text = []
        self._log_modes = []
        self._log_colours = []
        self.state = state.INITIAL
        self._queries = {}
        self._query_counter = 0
        self.session = session

    def run(self, input_queue):
        """
        Run command which is overriden by inheriting classes. The run function defines the action of the command through
        a series of actions which interact with the input queue.
        :param input_queue: Queue that is used by the function to interact with inputs from the display.
        :return: None
        """
        pass

    def get_log_data(self):
        """
        Return all the log data and clear the buffers.
        :return: Log data of the form (logs, log_modes, log_colors)
        """
        ret_logs = [x for x in self._log_text]
        ret_modes = [x for x in self._log_modes]
        ret_colours = [x for x in self._log_colours]
        self._log_text = []
        self._log_modes = []
        self._log_colours = []
        return ret_logs, ret_modes, ret_colours

    def log(self, text, mode='w', colour=1):
        """
        Log something that will be passed back to the display for the user to see.
        :param text: Text to be fed back to the user.
        :param mode: Mode of the log: w - write, a - append
        :param colour: Colour of the log entry. Colours are an integer value consistent with those defined in display.
        :return: None
        """
        if type(text) is not str:
            raise TypeError(f'Expected text for the log, instead got type:{type(text)}.')
        self._log_text.append(text)
        self._log_modes.append(mode)
        self._log_colours.append(colour)

    def register_query(self, query_string, state=None, mode='w', get_item=True, func=None):
        """
        Register a query with the command that will be used during the interaction. Registered queries are added to the
        queries list.
        :param query_string: The string to be displayed to the user.
        :param state: The state to be occupied by the end of the query.
        :param mode: Logging mode to be applied.
        :param get_item: Whether the query should expect to receive an item during this query.
        :param func: A function that may be called during this query.
        :return: None
        """
        new_idx = len(self._queries.keys())
        self._queries[new_idx] = (query_string, state, mode, get_item, func)


class StatusQuery(BaseCommand):
    """
    Query used to get the status of the different parts of the manager process.
    """
    def __init__(self, session):
        super().__init__(session=session)

        # iteration counter to keep track of the interaction
        self.query_iters = 0
        self.rjust_width = 25  # string width

        # register the status query
        self.register_query('', state.FINISHED, get_item=False, func=self.get_status)

        # hold the responses
        self.query_responses = {}

    def run(self, input_queue):
        qs, new_state, mode, get_item, func = self._queries[self.query_iters]

        # if we need an item grab it
        if get_item:
            item = input_queue.get()
            self.query_responses[self.query_iters - 1] = item
            self.log(item, mode='a')

        # log whatever is needed
        if qs != '':
            self.log(qs.rjust(self.rjust_width, ' '), mode)

        # run a function from the class
        if func is not None:
            func()

        # set the state
        if new_state is not None:
            self.state = new_state

        self.query_iters += 1
        return self.state

    def get_status(self):
        """
        Get the status of the manager print it to the display
        :return: None
        """
        default_str = ['Spooler Ready: '.rjust(self.rjust_width) + f'{self.session.manager.spooler_ready}',
                       'Spooler Configured: '.rjust(self.rjust_width) + f'{self.session.manager.spooler_configured}',
                       'Learners Initialised: '.rjust(self.rjust_width) + f'{self.session.manager.learners_initialised}',
                       ]

        # print the status of the manager for all to see
        self.log('Status'.center(100, '-'))
        self.log('-' * 100)
        for ds in default_str:
            self.log(ds)

        # figure out if the threads have started
        opt_thread, _ = self.session.manager.threads.get('optimise', (None, None))
        opt_thread_alive = 'Undefined'
        if opt_thread is not None:
            opt_thread_alive = opt_thread.is_alive()
        self.log('Optimise Thread Alive: '.rjust(self.rjust_width) + f'{opt_thread_alive}')
        self.log('-' * 100)

        # finish up
        self.state = state.FINISHED


class BestQuery(BaseCommand):
    """
    Query to list the currently known best cost and set of parameters.
    """
    def __init__(self, session):
        super().__init__(session=session)

        # iteration counter to keep track of the interaction
        self.query_iters = 0
        self.rjust_width = 22  # string width

        # hold the responses
        self.query_responses = {}

    def run(self, input_queue):
        """
        Pull the best cost and parameter sets from the current manager session and display them appropriately. Also
        handles appropriate formatting of the values.
        :param input_queue: Queue that is used by the function to interact with inputs from the display.
        :return: None
        """
        self.log('Best'.center(100, '-'))
        self.log('-' * 100)

        # format the current parameters to the best of our ability
        costs, params = self.session.manager.get_costs_params()
        if len(costs) > 0:
            self.log('Best Parameters:', colour=2)
            best_idx = np.argmin(costs)
            params = params[best_idx]

            # format options
            array_width = 5
            format_string = ' {: 1.3e}'

            param_string = '[ '
            for idx, val in enumerate(params):
                if not idx % array_width and idx:
                    self.log(param_string)
                    param_string = '  '
                param_string += format_string.format(val)

            self.log(param_string + '  ]')
            self.log(f'Best Cost:', colour=2)
            self.log('{:5.3f}'.format(costs[best_idx]))
        else:
            self.log('Costs and parameter lists are currently empty.', colour=2)

        # finish
        self.state = state.FINISHED
        self.log('-' * 100)


class PauseQuery(BaseCommand):
    """
    Pauses the current operations of the optimiser by setting the pause flag in the processes.
    """
    def __init__(self, session):
        super().__init__(session=session)

        # hold the responses
        self.query_responses = {}

    def run(self, input_queue):
        """
        Pause the manager session by calling on the pause_event which is thread safe. Logs to the display.
        :param input_queue: Queue that is used by the function to interact with inputs from the display.
        :return: None
        """
        self.log('Pausing execution ... ', colour=2)

        self.session.manager.pause_event.set()
        while not self.session.manager.paused.is_set():
            sleep(0.1)

        self.log('-'*100)
        self.log('Manager execution paused.', colour=2)

        self.state = state.FINISHED


class ResumeQuery(BaseCommand):
    """
    Resumes the processes after pause has been called on them.
    """
    def __init__(self, session):
        super().__init__(session=session)

        # hold the responses
        self.query_responses = {}

    def run(self, input_queue):
        """
        Resume the manager session by clearing the pause_event and paused event which is thread safe. Logs to the
        display.
        :param input_queue: Queue that is used by the function to interact with inputs from the display.
        :return: None
        """
        self.log('Resuming Execution ... ', colour=2)

        self.session.manager.paused.clear()
        self.session.manager.pause_event.clear()

        self.log('-'*100)
        self.log('Manager execution paused.', colour=2)

        self.state = state.FINISHED

class Help(BaseCommand):
    """
    Returns the list of possible commands that the user may call through the interface.
    """
    def run(self, input_queue):
        self.log('-' * 100)
        self.log('Commands:')
        command_list = ', '.join(registered_commands.keys())
        self.log(command_list)
        self.log('-' * 100)

        self.state = state.FINISHED


registered_commands = {"status": StatusQuery,
                       "best": BestQuery,
                       "pause": PauseQuery,
                       "resume": ResumeQuery,
                       "help": Help
                       }
