"""

Author: Tranter Tech
Date: 2024
"""
import json

from comms.TCP import FIFO
import logging
import session
from threading import Thread, Event
from queue import Queue, Empty
from time import sleep


class Manager:
    def __init__(self, server_instance):
        # server used for communications
        self.server = server_instance

        # define some FIFOs
        self.fifos = {'spooler': FIFO()}
        self.server.register_FIFO('spooler', self.fifos['spooler'])

        # define the threads that will be used by the manager
        self.threads = {'spooler': (Thread(target=self._spooler_thread), Queue()),
                        'optimise': (Thread(target=self._optimise), Queue())
                        }
        self.manager_halt = Event()
        self.spooler_ready = False
        self.spooler_configured = False
        self.learners_initialised = False
        self.learner_ids = None
        self.run_queue = Queue()

        # internal memory
        self._memory = {}
        self._optimisation_config = {}
        self._initialise_memory()
        self.interface = None
        self.interface_args = None

        # define the log
        self.log = logging.getLogger('Manager')
        self.log.addHandler(session.display_log_handler)

    def _initialise_memory(self):
        # live memory components that get updated as we go
        # (not thread safe, should only be accessed) in the manager scope by the manager
        self._memory['costs'] = []
        self._memory['parameters'] = []
        self._memory['bounds'] = []
        self._memory['run_number'] = 0
        self._memory['run_order'] = []
        self._memory['best_parameters'] = []
        self._memory['best_cost'] = None
        self._memory['optimisation_ok'] = False

        # static configurations for the optimisation (defaults) can be overridden by the user during the initialisation.
        self._optimisation_config['bound_restriction'] = 0.05
        self._optimisation_config['initial_count'] = 100
        self._optimisation_config['learner_number'] = 3
        self._optimisation_config['halt_number'] = 500
        self._optimisation_config['bounds'] = []

    def _initialise_connections(self):
        # start the server listening
        self.server.start_listening()

        # start the spooler
        self.log.info(f'Starting thread:[spooler] ...')
        thread, _ = self.threads['spooler']
        thread.start()

    def _spooler_config(self, spooler_fifo):
        # let the spooler know we're ready to go
        spooler_fifo.send('<CONF>')

        while not self.manager_halt.is_set():
            try:
                data = spooler_fifo.read(block=True, timeout=0.1)
                if data == '<CONF>':
                    self.log.debug('Spooler ready to accept configuration.')
                    break
                else:
                    self.log.warning(f'Expected <CONF> but got {data}.')
            except Empty:
                pass

        config_dict = {'bound_restriction': self._optimisation_config['bound_restriction'],
                       'learner_number': self._optimisation_config['learner_number'],
                       'bounds': self._optimisation_config['bounds'],
                       'initial_count': self._optimisation_config['initial_count']
                       }
        dict_string = json.dumps(config_dict)
        spooler_fifo.send(dict_string)

        while not self.manager_halt.is_set():
            try:
                data = spooler_fifo.read(block=True, timeout=0.1)
                if data == '<CA>':
                    self.log.debug('Spooler configuration complete.')
                    break
                else:
                    self.log.warning(f'Expected <CA> but got {data}.')
            except Empty:
                pass

    def _check_ready(self, spooler_fifo):
        data = ''
        try:
            data = spooler_fifo.read(block=True, timeout=0.1)
            self.log.debug(f'Spooler sent: {data}')
        except Empty:
            pass

        if data == '<READY>':
            return True
        else:
            return False

    def _spooler_thread(self):
        spooler_fifo = self.fifos['spooler']
        ready_sent = False
        self.spooler_configured = False
        _, manager_fifo = self.threads['spooler']
        while not self.manager_halt.is_set():

            # check the spooler is ready
            if not self.spooler_ready:
                if not ready_sent:
                    spooler_fifo.send('<READY>')
                    ready_sent = True

                self.spooler_ready = self._check_ready(spooler_fifo)
                continue

            # configure the spooler
            if not self.spooler_configured:
                self._spooler_config(spooler_fifo)
                self.spooler_configured = True

            # count the number of fails
            fail = 0

            # attempt to read from fifo
            data = ''
            try:
                data = spooler_fifo.read(block=False)
                self.log.debug(f'Spooler sent: {data}')
            except Empty:
                fail += 1

            # handle what should happen in the case of receiving something
            if data == '<CONF>':
                self._spooler_config(spooler_fifo)
            elif data == '<LI>':
                self.learners_initialised = True
            elif '<NP>' in data:
                self.run_queue.put(data[4:])
            elif '<QS>' in data:
                _, recv_fifo = self.threads['optimise']
                recv_fifo.put(data[4:])
            elif data == '<UR>':
                _, recv_fifo = self.threads['optimise']
                recv_fifo.put('<UR>')
            elif '<GL>' in data:
                # get the learner ids
                try:
                    self.learner_ids = json.loads(data[4:])
                except json.JSONDecodeError:
                    self.log.error('Couldn\'t get learner IDs.')

            # send any queued data
            try:
                send_str = manager_fifo.get(block=False)
                spooler_fifo.send(send_str)
            except Empty:
                fail += 1

            if fail > 1:
                # rate limit the loop
                sleep(0.01)

        self.log.info('Exited spooler thread.')

    def initialise_optimisation(self, options: dict, pre_load=None):
        """
        Initialise the options of the optimisation
        :param options: Dictionary which defines the optimisation, such as the bounds
        :param pre_load: Data that can be preloaded to kick off/pre-train the models
        :return:
        """
        self.log.info('Initialising optimisation ...')

        # grab all the values provided by the user for configuration
        config_keys = self._optimisation_config.keys()
        for key, value in options.items():
            if key.lower() in config_keys:
                self._optimisation_config[key.lower()] = value
            elif key.lower() not in ['interface', 'interface_args']:
                self.log.warning(f'Unknown key: {key}. Skipping from configuration process.')

        # pull out the interface separately
        self.interface = options.get('interface', None)
        self.interface_args = options.get('interface_args', None)
        if self.interface is None:
            self.log.error('No interface specified. Cannot start optimisation.')
            return 1

        # check whether we can start the optimisation
        config_ok = self.check_values()
        self._memory['optimisation_ok'] = config_ok

        if config_ok:
            self.log.info('Optimisation Configuration loaded:')
            self.log.info(f"    -Parameters: {len(self._optimisation_config['bounds'])}")
            for key, value in self._optimisation_config.items():
                self.log.info(f'    -{key}: {value}')

            # start the connections
            self._initialise_connections()

        # TODO - define the usage of pre_load

    def start_optimisation(self):
        if not self._memory['optimisation_ok']:
            self.log.error('Cannot start optimisation, provided values failed sanity check.')
            return False

        # check if the manager is ready
        self.log.info('Querying spooler status ...')
        _, manager_fifo = self.threads['spooler']

        # wait for the spooler to be ready
        while not self.manager_halt.is_set():
            if self.spooler_ready and self.spooler_configured:
                self.log.info('Spooler has registered ready and configured.')
                break
            else:
                sleep(0.1)

        # time to initialise the learners
        manager_fifo.put('<IL>')
        self.log.info('Starting learners ...')
        while not self.manager_halt.is_set():
            if not self.learners_initialised:
                sleep(0.1)
                continue
            break

        # get all the learner IDs
        manager_fifo.put('<GL>')
        while not self.manager_halt.is_set():
            if self.learner_ids is None:
                sleep(0.1)
                continue
            break
        self.log.debug(f'Spooler returned learner ids: {self.learner_ids}.')

        # start the optimisation loop
        self.log.info('Learners started. Starting interface...')

        # start the interface and determine whether it started
        interface_started = self.interface.run_initialisation(self.interface_args)
        if not interface_started:
            self.log.error('Interface could not start. Exiting.')
            self.close()
            return False

        self.log.info('Interface started. Beginning optimisation...')
        optimise_thread, _ = self.threads['optimise']
        optimise_thread.start()
        return True

    def check_values(self):
        # track the errors
        errors = []

        # check the bound restriction
        try:
            # can it be converted
            value = float(self._optimisation_config.get('bound_restriction', None))
            self._optimisation_config['bound_restriction'] = value

            # is it in the correct range
            if not 0.0 <= value <= 1.0:
                errors.append(f'bound_restriction:{value} should be in bounds [0, 1].')
        except ValueError as e:
            errors.append(f"Could not convert bound_restriction:"
                          f"{self._optimisation_config.get('bound_restriction', None)} to a float.")

        # check the learner number
        try:
            # can it be converted
            value = int(self._optimisation_config.get('learner_number', None))
            self._optimisation_config['learner_number'] = value

            # is it in the correct range
            if not 1 <= value:
                errors.append(f'learner_number:{value} should be >= 1.')
        except ValueError as e:
            errors.append(f"Could not convert learner_number:"
                          f"{self._optimisation_config.get('learner_number', None)} to an integer.")

        # check the initial count
        initial_count_ok = True
        try:
            # can it be converted
            value = int(self._optimisation_config.get('initial_count', None))
            self._optimisation_config['initial_count'] = value

            # is it in the correct range
            if not 0.0 <= value:
                errors.append(f'initial_count:{value} should be >= 0.')
                initial_count_ok = False
        except ValueError as e:
            errors.append(f"Could not convert initial_count:"
                          f"{self._optimisation_config.get('initial_count', None)} to an integer.")
            initial_count_ok = False

        # check the halt number
        try:
            # can it be converted
            value = int(self._optimisation_config.get('halt_number', None))
            self._optimisation_config['halt_number'] = value

            # is it in the correct range
            if value < 1:
                errors.append(f'Halt_number:{value} should be >= 1.')
            elif initial_count_ok:
                if self._optimisation_config['halt_number'] < self._optimisation_config['initial_count']:
                    errors.append(f"halt_number:{value} should be >= "
                                  f"initial_count:{self._optimisation_config['initial_count']}")

        except ValueError as e:
            errors.append(f"Could not convert halt_number:"
                          f"{self._optimisation_config.get('halt_number', None)} to an integer.")

        # check the bounds
        bound_type = type(self._optimisation_config.get('bounds', None))
        if bound_type is not tuple:
            errors.append(f'Expected tuple for bounds, got:{bound_type}.')
        else:
            if len(self._optimisation_config.get('bounds', None)) < 1:
                errors.append(f'Bounds should have a length of at least 1.')

            # make sure the bounds make sense
            for bound in self._optimisation_config.get('bounds', None):
                if type(bound) is not tuple:
                    errors.append(f'Bound:{bound} is not type:tuple with format (min, max).')

                # check there are a min and max value
                elif len(bound) != 2:
                    errors.append(f'Bound:{bound} is not of length 2.')

                try:
                    min = float(bound[0])
                    max = float(bound[1])

                    if not min < max:
                        errors.append(f'Bound:{bound} does not satisfy the condition min < max.')

                except ValueError as e:
                    errors.append(f'Error converting bound: {bound} -- {e.args[0]}')

        # list all the errors
        if len(errors) > 0:
            self.log.error('The following errors we found when checking the provided config:')
            for e in errors:
                self.log.error(f'    -{e}')
            return False

        # otherwise it's all good
        return True

    def _update_spooler(self, send_fifo, recv_fifo):
        # update the parameters to the spooler
        data_dict = {'parameters': self._memory['parameters'],
                     'costs': self._memory['costs']}
        data_string = json.dumps(data_dict)
        send_fifo.put('<UPDATE>')
        while not self.manager_halt.is_set():
            try:
                data = recv_fifo.get(block=True, timeout=0.1)
                if data == '<UR>':
                    send_fifo.put(data_string)
                    break
            except Empty:
                pass

    def _optimise(self):
        # communicate across comms
        _, send_fifo = self.threads['spooler']
        _, recv_fifo = self.threads['optimise']

        for lid in self.learner_ids:
            send_fifo.put(f'<QS:{lid}>')
            state = recv_fifo.get(True, timeout=10)
            self.log.debug(f'Learner {lid} returned state: {state}')

        # do the sampling first
        for run in range(self._optimisation_config['initial_count']):
            self._memory['run_number'] = run + 1

            # request a parameter
            send_fifo.put('<GET:SL1>')

            # look for a response
            next_params = None
            while not self.manager_halt.is_set():
                try:
                    data = self.run_queue.get(block=True, timeout=0.1)
                    data_dict = json.loads(data)
                    next_params = data_dict.get('params', None)
                    break
                except Empty:
                    pass
                except json.JSONDecodeError as e:
                    # if we failed to decode, try again unless it's occurred too many times
                    self.log.warning(f'Failed to decode JSON data: ({e.args})')
                    break

            if next_params is None:
                self.log.warning('Received no parameters. Skipping.')
                continue

            # call the interface to test
            cost = self.interface.run_parameters(next_params, args=self.interface_args)

            # update and log
            self._memory['costs'].append(cost)
            self._memory['parameters'].append(next_params)
            self._memory['run_order'].append('SL1')
            self.log.info(f"Sample {self._memory['run_number']} - cost: {cost}")

        # update the spooler parameters
        self._update_spooler(send_fifo, recv_fifo)

        self.log.info('Starting asynchronous predictions...')
        send_fifo.put('<SASYNC>')

        max_runs = self._optimisation_config['halt_number'] + self._optimisation_config['initial_count']
        start = self._optimisation_config['initial_count']
        for run in range(start, max_runs):
            self._memory['run_number'] = run + 1

            # look for a response
            next_params = None
            lid = None
            while not self.manager_halt.is_set():
                try:
                    data = self.run_queue.get(block=True, timeout=1)
                    json_data, lid = data.split('$$')
                    data_dict = json.loads(json_data)
                    next_params = data_dict.get('params', None)
                    break
                except Empty:
                    pass
                except json.JSONDecodeError as e:
                    # if we failed to decode, try again unless it's occurred too many times
                    self.log.warning(f'Failed to decode JSON data: ({e.args})')
                    break

            if next_params is None:
                self.log.warning('Received no parameters. Skipping.')
                continue

            # call the interface to test
            cost = self.interface.run_parameters(next_params, args=self.interface_args)

            # update and log
            self._memory['costs'].append(cost)
            self._memory['parameters'].append(next_params)
            self._memory['run_order'].append('SL1')
            self.log.info(f"Run {self._memory['run_number']} / {max_runs} - cost: {cost}")

            self._update_spooler(send_fifo, recv_fifo)
            send_fifo.put(f'<SR:{lid}>')

        self.log.warning('Finished Optimising')
        #
        #
        # # start looping for the total number of runs
        # for run in range(self._optimisation_config['halt_number']):
        #     self._memory['run_number'] = run + 1



    def close(self):
        self.server.close()

        self.manager_halt.set()
        for key, (thread, _) in self.threads.items():
            self.log.info(f'Waiting on [{key}] ...')
            if thread.is_alive():
                thread.join()

        self.log.info('Manager closed.')