"""

Author: Tranter Tech
Date: 2024
"""
from comms.TCP import ClientConnection, FIFO, Server
from utils.Tools import load_config, LearnerState
from threading import Thread, Event, Lock
from queue import Empty
from time import sleep
from copy import copy
import logging
import json
import session
from enum import Enum
from subprocess import Popen
import shlex


class Spooler:
    def __init__(self):
        self.log = logging.getLogger('Spooler')
        self.log.addHandler(session.display_log_handler)
        self.config = load_config(self.log)

        # set up the asynch server connection
        self.connection_fifo = FIFO()
        self.connection = ClientConnection('spooler', self.connection_fifo, self.config)

        # set up the server for the learners
        self.spooler_server_config = copy(self.config)
        self.spooler_server_config['tcp_port'] = self.spooler_server_config['spooler_tcp_port']
        self.learner_server = Server(self.spooler_server_config)
        self.learner_fifos = {}

        # thread for the spooler cycle
        self._spooler_cycle_thread = Thread(target=self._spooler_loop)
        self._halt_spooler = Event()

        # create a thread for asynchronous training
        self._asynch_prediction_thread = Thread(target=self._asynch_prediction)
        self._asynch_pause = Event()
        self._asynch_waiting = Event()
        self._asynch_waiting.set()

        # configuration dictionary for the spooler
        self._spooler_config = {'learner_number': None}
        self._spooler_configured = False

        # memory values to store
        self._memory = {'costs': [], 'parameters': []}
        self._memory_lock = Lock()

    def initialise_spooler(self):
        # start all the threads and services
        self.log.info('Initialising spooler ...')
        self.connection.comms_thread.start()
        self.learner_server.start_listening()
        self._spooler_cycle_thread.start()

    def _configure(self):
        """
        Configure the spooler with values from the manager communicated across the TCP socket
        :return: success (bool)
        """
        self.log.info('Getting configuration for spooler ...')
        self.connection_fifo.send('<CONF>')
        failures = 0

        # loop
        config_dictionary = {}
        while not self._halt_spooler.is_set():
            try:
                # read the data and convert it
                json_data = self.connection_fifo.read(block=True, timeout=0.1)
                config_dictionary = json.loads(json_data)
                break
            except Empty:
                pass
            except json.JSONDecodeError as e:
                # if we failed to decode, try again unless it's occurred too many times
                self.log.warning(f'Failed to decode JSON data: ({e.args})')
                self.connection_fifo.send('<CONF>')
                failures += 1
                if failures > 3:
                    return False

        # let the manager know we're happy
        self.connection_fifo.send('<CA>')

        # import all the values
        self.log.info('Configuring spooler with values:')
        for key, value in config_dictionary.items():
            self.log.info(f'    -Setting [{key}]: {value}')
            self._spooler_config[key] = value

        return True

    def _spooler_loop(self):
        self.log.info('Starting spooler loop ...')
        while not self._halt_spooler.is_set():
            # attempt to read from fifo
            try:
                data = self.connection_fifo.read(block=True, timeout=0.1)
                self.log.debug(f'Manager sent: {data}')

                # logic for each of the manager calls
                if data == '<READY>':
                    # respond ready if asked
                    self.connection_fifo.send('<READY>')
                    self.log.debug('Responding ready to manager.')
                elif data == '<CONF>':
                    self._spooler_configured = self._configure()
                elif data == '<IL>':
                    self._start_learners()
                elif data == '<QUIT>':
                    self.log.info('Received <QUIT>, shutting down...')
                    self.close()
                elif '<GET:' in data:
                    self.log.debug(f'Getting parameter {data}')
                    self._get_params(data)
                elif '<QS:' in data:
                    self.log.debug(f'Getting state {data}')
                    self._get_state(data)
                elif '<GL>' in data:
                    l_ids = self.learner_fifos.keys()
                    self.connection_fifo.send('<GL>' + json.dumps(list(l_ids)))
                elif data == '<SASYNC>':
                    self._asynch_prediction_thread.start()
                elif data == '<UPDATE>':
                    self._update_params()
                elif '<SR:' in data:
                    self._set_ready(data)
                elif data == '<BR>':
                    self._update_bounds()
                elif data == '<SAVE>':
                    self._save()

            except Empty:
                pass

        self.log.info('Spooler exited.')

    def _update_params(self):
        self.connection_fifo.send('<UR>')
        data_dict = self._read_json()

        self.log.debug('Got updated training data from manager.')
        if data_dict is not None:
            self._memory_lock.acquire()
            self._memory['parameters'] = data_dict.get('parameters', [])
            self._memory['costs'] = data_dict.get('costs', [])
            self._memory_lock.release()

    def _update_bounds(self):
        self.connection_fifo.send('<UR>')
        data_dict = self._read_json()

        if data_dict is not None:
            self._spooler_config['bounds'] = data_dict.get('bounds', self._spooler_config['bounds'])
            self.log.debug(f"Got updated bounds from manager. {self._spooler_config['bounds']}")

    def _read_json(self):
        """
        Read a JSON dictionary from the output
        :return: Data dictionary
        """
        # keep track of how many times we fail
        failures = 0

        # loop till completed or failed
        data_dictionary = None
        while not self._halt_spooler.is_set():
            try:
                # read the data and convert it
                json_data = self.connection_fifo.read(block=True, timeout=0.1)
                data_dictionary = json.loads(json_data)
                break
            except Empty:
                pass
            except json.JSONDecodeError as e:
                # if we failed to decode, try again unless it's occurred too many times
                self.log.warning(f'Failed to decode JSON data: ({e.args})')
                failures += 1
                if failures > 3:
                    break

        return data_dictionary

    def _spawn_learner(self, learner_id, cmd=None):
        # add all the relevant fifos
        fifo = FIFO()
        self.learner_fifos[learner_id] = fifo
        self.learner_server.register_FIFO(learner_id, fifo)

        # pull in the pre- and post-amble and construct
        pre_cmd = self._spooler_config['terminal_cmds'][0]
        post_cmd = self._spooler_config['terminal_cmds'][1]
        if cmd is None:
            cmd = f'python main.py --learner -i {learner_id[-1]}'
        cmd = f'{pre_cmd}{cmd}{post_cmd}'
        Popen(shlex.split(cmd))

        # Start the configuration for this learner
        self.learner_fifos[learner_id].send('<CONFIG>')

    def _configure_learner(self, learner_id):
        self.log.info(f'Configuring learner [{learner_id}].')
        # attempt to configure each learner
        config_sent = False
        while not self._halt_spooler.is_set():
            if not config_sent:
                try:
                    # acknowledge and send the
                    data = self.learner_fifos[learner_id].read(block=True, timeout=0.1)
                    if data == '<CA>':
                        self.log.debug(f'Sending {learner_id} config ...')

                        # dump the data dictionary and send it
                        dump_str = json.dumps({'n_inputs': len(self._spooler_config['bounds']),
                                               'bounds': self._spooler_config['bounds'],
                                               'initial_count': self._spooler_config['initial_count'],
                                               'learner_min_tol': self._spooler_config['learner_min_tol']
                                               })
                        self.learner_fifos[learner_id].send(dump_str)
                        config_sent = True
                    else:
                        self.log.warning(f'Expected <CA> from {learner_id}, got [{data}].')
                        break
                except Empty:
                    pass
            else:
                try:
                    # make sure it was received, otherwise log it
                    data = self.learner_fifos[learner_id].read(block=True, timeout=0.1)
                    if data == '<CR>':
                        self.log.debug(f'Learner {learner_id} received config.')
                        break
                    else:
                        self.log.warning(f'Expected <CR> from {learner_id}, got [{data}].')
                        break
                except Empty:
                    pass

    def _start_learners(self):
        # check we are ready
        if not self._spooler_configured:
            self.log.warning('Manager attempted to start learners before configuration.')
            self.connection_fifo.send('<ERROR>')
            return False

        self.log.info('Starting sampling learner ... ')
        # start with the sampler
        sampler_cmd = f'python main.py --learner --sampler -i 1'
        self._spawn_learner('SL1', cmd=sampler_cmd)
        self._configure_learner('SL1')

        n_learners = self._spooler_config.get('learner_number', None)
        if n_learners is None:
            n_learners = 1
            self.log.warning('Number of learners not found. Defaulting to 1.')

        self.log.info(f'Starting {n_learners} NN learners.')

        # sequentially load all the learners
        for i in range(1, n_learners+1):
            learner_id = f'NN{i}'
            self._spawn_learner(learner_id)

        # split this loop as tensorflow takes a bit to start up
        for i in range(1, n_learners+1):
            learner_id = f'NN{i}'
            self._configure_learner(learner_id)

        self.connection_fifo.send('<LI>')

    def _get_params(self, flag):
        flag_split = flag.split(':')
        if len(flag_split) < 2:
            self.log.error('No learner id found from flag.')
            self.connection_fifo.send('null')
            return False

        # get the ID
        learner_id = flag_split[-1][:-1]
        self.log.debug(f'Querying [{learner_id}] for params...')
        self.learner_fifos[learner_id].send('<GET>')

        while not self._halt_spooler.is_set():
            try:
                # acknowledge and send the
                data = self.learner_fifos[learner_id].read(block=True, timeout=0.1)
                self.connection_fifo.send('<NP>' + data)
                break
            except Empty:
                pass

    def _get_state(self, flag, return_state=False):
        flag_split = flag.split(':')
        if len(flag_split) < 2:
            self.log.error('No learner id found from flag.')
            self.connection_fifo.send('null')
            return False

        # get the ID
        learner_id = flag_split[-1][:-1]
        self.log.debug(f'Querying [{learner_id}] for state...')
        self.learner_fifos[learner_id].send('<QSTATE>')

        while not self._halt_spooler.is_set():
            try:
                # acknowledge and send the
                data = self.learner_fifos[learner_id].read(block=True, timeout=0.1)
                if not return_state:
                    self.connection_fifo.send('<QS>' + data)
                    break
                else:
                    return data
            except Empty:
                pass

    def _set_ready(self, flag):
        flag_split = flag.split(':')
        if len(flag_split) < 2:
            self.log.error('No learner id found from flag.')
            self.connection_fifo.send('null')
            return False

        # get the ID
        learner_id = flag_split[-1][:-1]

        lf = self.learner_fifos.get(learner_id, None)
        if lf is not None:
            self.log.debug(f'Setting learner {learner_id} to ready.')
            lf.send('<READY>')
            return True

    def _asynch_prediction(self):
        self.log.info('Kicking off asynchronous prediction cycle ... ')
        self._asynch_waiting.clear()
        while not self._halt_spooler.is_set():
            for key, fifo in self.learner_fifos.items():

                # pause if asked
                if self._asynch_pause.is_set():
                    self.log.debug('Found asynch pause. Waiting...')
                    self._asynch_waiting.set()
                    while self._asynch_pause.is_set():
                        sleep(0.1)

                if key == 'SL1':
                    continue

                try:
                    # only query if we're still waiting
                    if not fifo.data_ready():
                        fifo.send('<QSTATE>')

                    # get out the response
                    state = fifo.read(block=True, timeout=1)
                    ls = LearnerState(int(state))
                    if ls not in [LearnerState.BUSY, LearnerState.PREDICTED]:
                        self.log.debug(f'Learner {key} in state: {ls}')

                    if ls == LearnerState.READY:
                        # if we're ready get the latest update params
                        self._memory_lock.acquire()
                        update_dict = self._memory.copy()
                        update_dict['bounds'] = self._spooler_config['bounds']
                        data_string = json.dumps(update_dict)
                        self._memory_lock.release()

                        # let the learner know an update is coming
                        fifo.send('<UPDATE>')
                        response = fifo.read(block=True, timeout=1)
                        if response == '<CA>':
                            self.log.debug(f'Sending data string to learner.')
                            fifo.send(data_string)

                        response = fifo.read(block=True, timeout=5)
                        if response == '<CA>':
                            self.log.debug(f'Values updated for {key}.')

                    elif ls == LearnerState.TRAIN:
                        self.log.debug(f'Starting {key} on training.')
                        fifo.send('<TRAIN>')

                    elif ls == LearnerState.MINIMISING:
                        self.log.debug(f'Starting {key} minimising.')
                        fifo.send('<MIN>')

                    elif ls == LearnerState.PREDICT:
                        self.log.debug(f'Getting prediction from {key}...')
                        fifo.send('<GET>')
                        data = fifo.read()
                        self.connection_fifo.send('<NP>' + data + f'$${key}')

                except Empty:
                    pass

        self.log.info('Asynchronous predictions finished.')
        self._asynch_waiting.set()  # set to allow saving afterwards

    def _save(self):
        self.connection_fifo.send('<SAVE>')
        self.log.info('Gathering data from learners ... ')

        self._asynch_pause.set()
        self.log.debug('Waiting on asynch thread ...')
        while not self._asynch_waiting.is_set():
            sleep(0.1)

        for key, fifo in self.learner_fifos.items():
            self.log.debug(f'Sending {key} save command.')
            fifo.send('<SAVE>')

        returned_keys = []
        learner_data = []
        while not self._halt_spooler.is_set():
            if len(learner_data) == len(self.learner_fifos):
                self.log.info('Got all learner data.')
                break

            for key, fifo in self.learner_fifos.items():
                if key in returned_keys:
                    continue
                try:
                    data = fifo.read(block=True, timeout=1)
                    learner_data.append(data)
                    returned_keys.append(key)
                    self.log.debug(f'Got data from {key}.')
                except:
                    pass

        # send back all the data we got
        for data, key in zip(learner_data, returned_keys):
            self.connection_fifo.send(f'<LD>{key}={data}')

        self.connection_fifo.send('<FIN>')

        self._asynch_waiting.clear()
        self._asynch_pause.clear()

    def close(self):
        self.log.info('Shutting down spooler ...')

        # send the quit command to all learners
        for key, fifo in self.learner_fifos.items():
            fifo.send('<QUIT>')

        self.connection.close()
        self.learner_server.close()
        self._halt_spooler.set()

        if self._asynch_prediction_thread.is_alive():
            self.log.debug('Waiting on asynch thread ...')
            self._asynch_prediction_thread.join()


