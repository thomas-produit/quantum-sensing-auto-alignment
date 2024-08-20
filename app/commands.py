"""
This file is to hold the relevant commands that are used by the program. Each command should be designated as a class
which inherits from the base command class.
"""
from enum import Enum
from datetime import datetime


class state(Enum):
    INITIAL = 0
    RUNNING = 1
    FINISHED = 2


class LogEvent(Enum):
    ENTERING = 0
    LEAVING = 1


class BaseCommand:
    def __init__(self, session=None):
        self._log_text = []
        self._log_modes = []
        self.state = state.INITIAL
        self._queries = {}
        self._query_counter = 0
        self.session = session

    def run(self, input_queue):
        pass

    def get_log_data(self):
        ret_logs = [x for x in self._log_text]
        ret_modes = [x for x in self._log_modes]
        self._log_text = []
        self._log_modes = []
        return ret_logs, ret_modes

    def log(self, text, mode='w'):
        if type(text) is not str:
            raise TypeError(f'Expected text for the log, instead got type:{type(text)}.')
        self._log_text.append(text)
        self._log_modes.append(mode)

    def register_query(self, query_string, state=None, mode='w', get_item=True, func=None):
        new_idx = len(self._queries.keys())
        self._queries[new_idx] = (query_string, state, mode, get_item, func)


class TestQuery(BaseCommand):
    def __init__(self, session):
        super().__init__(session=session)

        # iteration counter to keep track of the interaction
        self.query_iters = 0
        self.rjust_width = 22  # string width

        self.register_query('Name: ', state.RUNNING, get_item=False)
        self.register_query('', func=self.check_input)

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

    def check_input(self):
        """
        Check if the input is correct or not and act accordingly
        :return:
        """
        # Depending on the query position we should do something
        responses = max(self.query_responses.keys())
        self.log('Query Finished')
        self.state = state.FINISHED


class Help(BaseCommand):
    def run(self, input_queue):
        self.log('-' * 100)
        self.log('Commands:')
        command_list = ', '.join(registered_commands.keys())
        self.log(command_list)
        self.log('-' * 100)

        self.state = state.FINISHED


registered_commands = {"new": TestQuery,
                       "help": Help
                       }
