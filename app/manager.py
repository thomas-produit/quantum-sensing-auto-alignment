"""

Author: Tranter Tech
Date: 2024
"""
import json

from comms.TCP import FIFO
import logging
import session
from threading import Thread, Event
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Queue, Empty
from time import sleep
import numpy as np
from datetime import datetime
import os
import scipy.cluster as scl
from scipy.signal import savgol_filter

# only import the graphical part if we can
_GRAPHICAL = True
_PRE_ERRORS = []
try:
    from app.graphical import RealTimePlot, _DEFAULT_CONFIG
except ImportError as e:
    _PRE_ERRORS.append(f"Couldn't load graphical library: {e.args}.")
    _GRAPHICAL = False


class Manager:
    def __init__(self, server_instance, runtime_config=None):
        # server used for communications
        self.server = server_instance

        # config imported during the initial execution
        if runtime_config is None:
            self._runtime_config = {}
        else:
            self._runtime_config = runtime_config

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

        # used to handle the final saving event
        self.save_event = Event()

        # internal memory
        self._memory = {}
        self._optimisation_config = {}
        self._heuristic_tracker = None
        self._initialise_memory()
        self.interface = None
        self.interface_args = None

        # graphical process to be used for plotting
        self._graphical_proc = None
        self._plot_queue = MPQueue()
        if _GRAPHICAL:
            self._graphical_proc = Process(target=RealTimePlot, args=(_DEFAULT_CONFIG, self._plot_queue))
            self._scale_func = None

        # define the log
        self.log = logging.getLogger('Manager')
        self.log.addHandler(session.display_log_handler)
        for err in _PRE_ERRORS:
            self.log.warning(err)

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
        self._memory['graphical'] = _GRAPHICAL

        # static configurations for the optimisation (defaults) can be overridden by the user during the initialisation.
        self._optimisation_config['bound_restriction'] = 0.05
        self._optimisation_config['initial_count'] = 100
        self._optimisation_config['learner_number'] = 3
        self._optimisation_config['halt_number'] = 500
        self._optimisation_config['bounds'] = []
        self._optimisation_config['learner_min_tol'] = 1E-4

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
                       'initial_count': self._optimisation_config['initial_count'],
                       'data_dir': self._optimisation_config['data_dir'],
                       'terminal_cmds': self._runtime_config.get('terminal_cmds', ['', '']),
                       'learner_min_tol': self._optimisation_config['learner_min_tol']
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
            elif data == '<SAVE>':
                self._save_models(spooler_fifo)

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

        # throw an error if the options are not a dictionary
        if type(options) is not dict:
            raise TypeError(f'Expected a dictionary, instead received type:<{type(options)}>.')

        self.log.debug('Checking if data directory exists')
        data_loc = options.get('data_directory', './data').strip('/')
        if not os.path.exists(data_loc):
            try:
                os.mkdir(data_loc)
                self.log.info(f'Created data directory @\'{data_loc}\'.')
            except PermissionError as e:
                self.log.warning(f'Could not create data directory @\'{data_loc}\'. Defaulting to ./data')
                self.log.debug(f'Data directory error {e.args}.')
                data_loc = './data'
                if not os.path.exists(data_loc):
                    os.mkdir(data_loc)

        # check we have permission to write
        write_access = os.access(data_loc, os.W_OK)
        if not write_access:
            self.log.error(f'Write permissions not available for {data_loc}. Specify new location.')
            return 1

        # save the location for later use
        self._optimisation_config['data_dir'] = data_loc

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

        self.interface.plot_queue = self._plot_queue

        # check whether we can start the optimisation
        config_ok = self.check_values()
        self._memory['optimisation_ok'] = config_ok

        if config_ok:
            self.log.info('Optimisation Configuration loaded:')
            self.log.info(f"    -Parameters: {len(self._optimisation_config['bounds'])}")
            for key, value in self._optimisation_config.items():
                self.log.info(f'    -{key}: {value}')

            self._memory['bounds'] = self._optimisation_config['bounds']

            # expect to get a dict with keys: [costs, params], where costs is N*1 and params is N*K arrays.
            pre_load_ok = True
            if pre_load is not None:
                if type(pre_load) is not dict:
                    self.log.error('Pre-loaded data is of the wrong type.')
                    pre_load_ok = False
                else:
                    costs = pre_load.get('costs', None)
                    params = pre_load.get('params', None)

                    if costs is None:
                        self.log.error('No costs key located in the pre_load.')
                        pre_load_ok = False
                    elif params is None:
                        self.log.error('No params key located in the pre_load.')
                        pre_load_ok = False
                    else:
                        assert type(costs) is list, "Expected type list for the costs pre-load."
                        assert type(params) is list, "Expected type list for the params pre-load."

                        costs = np.array(costs)
                        params = np.array(params)

                        r, c = costs.shape
                        rp, cp = params.shape
                        if c != 1:
                            self.log.error(f'Expected a vector for the costs not shape {(r, c)}')
                            pre_load_ok = False
                        if rp != r:
                            self.log.error(f'Costs and params should have same length: ({r}, {rp}).')
                            pre_load_ok = False
                        if len(self._memory['bounds']) != cp:
                            self.log.error(f'Params should have the same length as the bounds: {params.shape}')
                            pre_load_ok = False

                        if pre_load_ok:
                            self.log.info('Loading the pre-load data.')
                            for c in costs:
                                self._memory['costs'].append(c)
                            for p in params:
                                self._memory['parameters'].append(list(p))

            # start the graphical interface if we need
            if self._memory.get('graphical', False):
                self._graphical_proc.start()

                min, max = list(zip(*self._memory['bounds']))
                self._scale_func = lambda X: (np.array(X) - np.array(min)) / (np.array(max) - np.array(min))

                # load the current values into the graphical proc
                for cost, params in zip(self._memory['costs'], self._memory['parameters']):
                    self._update_plot(params, cost)

            # start the connections
            self._initialise_connections()

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
            # elif initial_count_ok:
            #     if self._optimisation_config['halt_number'] < self._optimisation_config['initial_count']:
            #         errors.append(f"halt_number:{value} should be >= "
            #                       f"initial_count:{self._optimisation_config['initial_count']}")

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
                     'costs': self._memory['costs'],
                     'bounds': self._memory['bounds']}
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

    def _bound_restrict(self, send_fifo, recv_fifo, restrict, best_point=None):
        # get the new bounds to send
        new_bounds = []
        if restrict:
            factor = self._optimisation_config['bound_restriction']
            new_bound_span = [(b[1] - b[0]) * factor for b in self._memory['bounds']]
            for (bmin, bmax), span, param_value in zip(self._memory['bounds'], new_bound_span, best_point):
                new_bmin = max(param_value - (span / 2), bmin)
                new_bmax = min(param_value + (span / 2), bmax)
                new_bounds.append((new_bmin, new_bmax))
        else:
            new_bounds = self._memory['bounds']

        # update the heuristics as well
        self._heuristic_tracker.update_bounds(new_bounds)

        send_fifo.put('<BR>')
        while not self.manager_halt.is_set():
            try:
                data = recv_fifo.get(block=True, timeout=0.1)
                if data == '<UR>':
                    data_dict = {'bounds': new_bounds}
                    send_fifo.put(json.dumps(data_dict))
                    break
            except Empty:
                pass

        self.log.debug('Updated spooler bounds.')

    def _optimise(self):
        # communicate across comms
        _, send_fifo = self.threads['spooler']
        _, recv_fifo = self.threads['optimise']

        for lid in self.learner_ids:
            send_fifo.put(f'<QS:{lid}>')
            state = recv_fifo.get(True, timeout=10)
            self.log.debug(f'Learner {lid} returned state: {state}')

        # Variables for keeping track of the heuristics
        runs_since_improvement = 0
        bounds_restricted = False

        # -----------------------------------------------
        # do the sampling first
        # -----------------------------------------------
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

            # update all the running bests
            if self._memory['best_cost'] is None:
                self._memory['best_cost'] = cost
                self._memory['best_parameters'] = next_params
                self._update_plot(next_params, cost, update_best=True)
            elif cost < self._memory['best_cost']:
                self.log.info(f'New best cost found: {cost}', extra={'colour': 4})
                self._update_plot(next_params, cost, update_best=True)
                self._memory['best_cost'] = cost
                self._memory['best_parameters'] = next_params
            else:
                # plot the new parameters and cost
                self._update_plot(next_params, cost)

        # feedback the best so far
        self.log.info(f"Sampling finished. Best cost observed: {self._memory['best_cost']}")

        # update the spooler parameters
        self._update_spooler(send_fifo, recv_fifo)

        self.log.info('Starting asynchronous predictions...')
        send_fifo.put('<SASYNC>')

        max_runs = self._optimisation_config['halt_number'] + self._optimisation_config['initial_count']
        start = self._optimisation_config['initial_count']

        # define the heuristic tracker for bumping
        self._heuristic_tracker = HeuristicTracker(self._memory['bounds'],
                                                   self._memory['best_parameters'],
                                                   self.log)
        self._heuristic_tracker.last_params = np.array(self._memory['best_parameters'])

        # -----------------------------------------------
        # Neural net time
        # -----------------------------------------------
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

            # bump the parameters if necessary
            self._heuristic_tracker.update_costs_params(self._memory['costs'], self._memory['parameters'])
            next_params = self._heuristic_tracker.modify_parameters(next_params)

            # call the interface to test
            cost = self.interface.run_parameters(next_params, args=self.interface_args)

            # update all the running bests
            if cost < self._memory['best_cost']:
                self.log.info(f'New best cost found: {cost}', extra={'colour': 4})
                self._memory['best_cost'] = cost
                self._memory['best_parameters'] = next_params
                runs_since_improvement = 0
                self._heuristic_tracker.update_best(next_params)

                # plot the new parameters and cost
                self._update_plot(next_params, cost, update_best=True)
            else:
                runs_since_improvement += 1
                # plot the new parameters and cost
                self._update_plot(next_params, cost)

            # kick off a local on the next iteration if we need to
            if runs_since_improvement >= 10:
                runs_since_improvement = 0

                # restrict the bounds to do a local search
                if not bounds_restricted:
                    self.log.info('Restricting bounds for local search.', extra={'colour': 2})
                    self._bound_restrict(send_fifo, recv_fifo, True, self._memory['best_parameters'])
                else:
                    self.log.info('Returning bounds for exploration.', extra={'colour': 2})
                    self._bound_restrict(send_fifo, recv_fifo, False, None)

                # flip the boolean
                bounds_restricted = not bounds_restricted

            # update and log
            self._memory['costs'].append(cost)
            self._memory['parameters'].append(next_params)
            self._memory['run_order'].append(lid)
            self.log.info(f"Run {self._memory['run_number']} / {max_runs} - cost: {cost}")

            self._update_spooler(send_fifo, recv_fifo)
            send_fifo.put(f'<SR:{lid}>')

        self.log.info('Finished Optimising',  extra={'colour': 4})
        self.log.info(f"Best Cost: {self._memory['best_cost']}", extra={'colour': 4})
        self.log.info(f"Best Params: {self._memory['best_parameters']}", extra={'colour': 4})

    def _update_plot(self, params, cost, update_best=False):
        # if we can plot, then plot
        if self._memory.get('graphical', False):
            self._plot_queue.put(('append', {'y': [cost]}, 'cost_hist'))
            self._plot_queue.put(('replace', {'x': range(len(params)),
                                              'y': params,
                                              'args': {'pen': None, 'symbolBrush': 'b'}},
                                  'nets'
                                  ))

            # plot with respect to the center
            N = float(len(params))
            dist_0 = np.sum(np.square(self._scale_func([0] * int(N)) - self._scale_func(params))) / N
            self._plot_queue.put(('append', {'y': [dist_0]}, 'dist_0'))

            # plot with respect to the best know params
            distance = np.square(np.array(self._memory['best_parameters']) - np.array(params)).sum()
            self._plot_queue.put(('append', {'y': [distance]}, 'dist'))

            # only update the best if we need to
            if update_best:
                self._plot_queue.put(('replace', {'x': range(len(params)),
                                                  'y': params,
                                                  'args': {'pen': None, 'symbolBrush': 'b'}},
                                     'best'
                                      ))

    def _save_models(self, spooler_fifo):
        while not self.manager_halt.is_set():
            try:
                data = spooler_fifo.read(block=True, timeout=0.1)
                if data == '<SAVE>':
                    self.log.debug('Spooler ready to send model info ...')
                elif '<LD>' in data:
                    key, data = data.split('=')
                    self.log.debug(f'Learner {key} sent: {data}')
                elif data == '<FIN>':
                    self.log.debug('All learner data collected.')
                    break
            except Empty:
                pass

        # clear the event so we can exit
        self.save_event.set()

    def save(self):
        datetime_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
        save_loc = self._optimisation_config.get('data_dir', '/NOT_SPECIFIED/')
        self.log.info('Saving data ... ', extra={'color': 2})

        # create a new_dir
        new_dir = f'{save_loc}/{datetime_str}_optim'
        os.mkdir(new_dir)

        # save the run specific data
        self.log.debug('Saving run data ...', extra={'color': 2})
        with open(f'{new_dir}/data_{datetime_str}.json', 'w') as f:
            json.dump(self._memory, f)

        # get the FIFO to communicate with the spooler
        # _, manager_fifo = self.threads['spooler']
        # self.log.info('Getting model data ...')
        # manager_fifo.put('<SAVE>')

        # # wait for the saving to finish
        # while not self.save_event.is_set():
        #     sleep(0.1)

        self.log.info('Saving models complete.')

    def close(self):
        # save the data
        self.save()

        # close the server
        self.server.close()

        # tell the manager threads to halt and wait for them
        self.manager_halt.set()
        for key, (thread, _) in self.threads.items():
            self.log.info(f'Waiting on [{key}] ...')
            if thread.is_alive():
                thread.join()

        self.log.info('Manager closed.')

        # wait for the graphical process if it exists
        if self._graphical_proc is not None:
            self.log.info('Waiting on graphical process')
            self._graphical_proc.terminate()
            try:
                self._graphical_proc.join()
            except ValueError:
                pass

        self.log.info('Process halted :)', extra={'colour': 4})

class HeuristicTracker:
    def __init__(self, bounds, best_params, log):
        mins, maxs = list(zip(*bounds))
        self.bounds = bounds
        self.scale_func = lambda X: (np.array(X) - np.array(mins)) / (np.array(maxs) - np.array(mins))
        self.num_params = len(bounds)

        self.last_params = None
        self.best_params = best_params

        # get the bump starts for each parameter
        self.spans = (np.array(maxs) - np.array(mins))
        ooms = np.round(np.log(self.spans) / np.log(10))
        self.bump_starts = 10 ** (ooms - 3)
        self.current_bumps = np.array([bp for bp in self.bump_starts])

        self.bump_incrementer = 0
        self.bump_method = 0
        self.log = log

        self.runs_without_increase = 0

        self.costs = None
        self.params = None

    def update_costs_params(self, costs, params):
        """
        Update the cost and parameter arrays, so we can use them in the bumping
        :param costs:
        :param params:
        :return:
        """
        self.costs = np.array(costs)
        self.params = np.array(params)

    def modify_parameters(self, next_params):
        next_params = np.array(next_params)
        distance = np.sum(
            np.square(self.scale_func(self.best_params) - self.scale_func(next_params))) / self.num_params
        distance_last = np.sum(
            np.square(self.scale_func(self.last_params) - self.scale_func(next_params))) / self.num_params

        bumped = False
        self.last_params = next_params
        self.runs_without_increase += 1

        if self.runs_without_increase > 3 and (distance < 0.03 or distance_last < 0.05):
            bumped = True

            # increment the bump if we need to
            self.runs_without_increase = 0
            if self.bump_incrementer > 3:
                self.bump_incrementer = 0

                # reset the bump scale if it's too big
                if self.current_bumps[0] >= (self.bump_starts[0] * 100):
                    self.log.debug('Resetting bump scale.')
                    self.current_bumps = np.array([bp for bp in self.bump_starts])
                else:
                    self.current_bumps *= 2
            else:
                self.bump_incrementer += 1

            if self.bump_method == 0:
                # Method 1: bump the parameters randomly
                offset = self.spans * np.random.uniform(-0.1, 0.1, 1)
                polarity_mask = np.random.randint(0, 2, self.num_params)
                polarity_mask[polarity_mask == 0] = -1

                self.log.info(f'Applying bump: {self.current_bumps} - {offset}',  extra={'colour': 4})
                next_params += np.multiply(polarity_mask, self.current_bumps + offset)

            elif self.bump_method == 1:
                # Method 2: Cluster the elements
                if self.costs is None or self.params is None or len(self.costs) < 10:
                    # skip if we haven't defined these yet
                    self.log.debug('Threshold or parameters not met for clustering.')
                    return list(self.clip_next(next_params))

                # sort all the elements and mask the top 30% of costs
                csorted = sorted(self.costs)
                cmax = csorted[len(self.costs) // 3]
                self.log.debug(f'C_max for clusters: {cmax}')
                mask = np.where(self.costs < cmax)

                # cluster the masked parameters
                clusters = scl.hierarchy.fclusterdata(self.params[mask], 0.75)
                cluster_max = np.max(clusters)
                self.log.debug(f'Cluster Max: {cluster_max}')

                # get the top cluster and params
                try:
                    c, cl, p = zip(*sorted(zip(self.costs[mask], clusters, self.params), key=lambda x: x[0]))
                    best_params = p[0]
                    best_cluster = cl[0]

                    cl_arr = np.array(cl)
                    cl_mask = np.where(cl_arr != best_cluster)
                    next_cluster_idx = np.random.randint(0, len(cl_arr[cl_mask]))
                    next_cluster_params = np.array(p)[cl_mask][next_cluster_idx]

                    direction = best_params - next_cluster_params
                    next_params = best_params - (direction*self.current_bumps)
                except Exception as err:
                    self.log.error(f'Failed to cluster: {err.args}')

                self.log.info(f'Applying cluster bump: {next_params}', extra={'colour': 4})

            elif self.bump_method == 2:
                # Method 3: Savgol Filter
                window_length = min(40, len(self.params) // 2)
                order = min(10, window_length - 1)

                bumps = np.zeros(self.num_params)
                for i in range(self.num_params):
                    xs = self.params[:, 0]
                    ys = self.costs[:]

                    zipped = sorted(zip(xs, ys), key=lambda x: x[0])
                    xs, ys = zip(*zipped)

                    filtered = savgol_filter(ys, window_length, order, axis=0)
                    bump = (next_params[i] - xs[np.argmin(filtered)])

                    next_params[i] -= bump
                    bumps[i] = bump

                self.log.info(f'Applying SF bump: {next_params}', extra={'colour': 4})

            # cycle the different types of bumps
            self.bump_method = (self.bump_method + 1) % 3

        next_params = self.clip_next(next_params)
        return list(next_params)

    def update_bounds(self, bounds):
        mins, maxs = list(zip(*bounds))
        self.bounds = bounds
        self.scale_func = lambda X: (np.array(X) - np.array(mins)) / (np.array(maxs) - np.array(mins))
        self.num_params = len(bounds)

        # get the bump starts for each parameter
        self.spans = (np.array(maxs) - np.array(mins))
        ooms = np.round(np.log(self.spans) / np.log(10))
        self.bump_starts = 10 ** (ooms - 3)
        self.current_bumps = np.array([bp for bp in self.bump_starts])

    def update_best(self, best_params):
        self.runs_without_increase = 0
        self.best_params = best_params

    def clip_next(self, next_params):
        for idx, ((bmin, bmax), param) in enumerate(zip(self.bounds, next_params)):
            next_params[idx] = min(bmax, max(param, bmin))

        return next_params
