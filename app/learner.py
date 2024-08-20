"""

Author: Tranter Tech
Date: 2024
"""
from enum import Enum
import session
from utils.Tools import load_config, LearnerState
import logging
from comms.TCP import FIFO, ClientConnection
from threading import Thread, Event
from queue import Queue, Empty
import tensorflow as tf
import json
import numpy as np
from time import sleep
from sklearn.preprocessing import StandardScaler
from multiprocessing.pool import ThreadPool
from multiprocessing import Pipe
import scipy.optimize as so


class BaseLearner:
    """
    Base learner class that other learners inherit.
    """

    def __init__(self, learner_id):
        self.id = learner_id
        self._memory = {'costs': [], 'parameters': []}
        self._state = LearnerState.PRE_INIT

        # get the logs and config
        self.log = logging.getLogger(f'Learner:[{self.id}]')
        self.log.addHandler(session.display_log_handler)
        self.config = load_config(self.log)

        # modify the port to be the correct port
        self.config['tcp_port'] = self.config['spooler_tcp_port']

        # set up the asynch server connection
        self.connection_fifo = FIFO()
        self.connection = ClientConnection(self.id, self.connection_fifo, self.config)
        self.learner_halt = Event()
        self.comm_thread = Thread(target=self._comm_cycle)

        # commands to be invoked in to loop
        self._commands = {}

        # register some basic commands
        self._register_command('<UPDATE>', self._update_values)
        self._register_command('<QUIT>', self.halt)
        self._register_command('<CONFIG>', self._get_config)
        self._register_command('<QSTATE>', self._get_state)
        self._register_command('<GET>', self._get_next_parameters)
        self._register_command('<TRAIN>', self.train)
        self._register_command('<MIN>', self.minimise)
        self._register_command('<READY>', self.ready)


    def _register_command(self, cmd_flag, function):
        """
        Register a command and function call for the comms cycle.
        :param cmd_flag: Flag to come from the spooler server to kick off the command
        :param function: Function to be called with the matching flag
        :return: None
        """
        self._commands[cmd_flag.lower()] = function

    def _comm_cycle(self):
        while not self.learner_halt.is_set():

            # attempt to read from fifo
            try:
                data = self.connection_fifo.read(block=True, timeout=0.01)

                # run the relevant code
                for key, function in self._commands.items():
                    if key == data.lower():
                        if key.lower() != '<qstate>':
                            self.log.debug(f'Matched {key}: calling function [{function.__name__}].')
                        function()

            except Empty:
                pass

        self.log.info(f'Learner {self.id} exited.')

    def _initialise(self):
        self.connection.comms_thread.start()
        self.comm_thread.start()
        self._state = LearnerState.INIT

    def _read_json(self):
        """
        Read a JSON dictionary from the output
        :return: Data dictionary
        """
        # keep track of how many times we fail
        failures = 0

        # loop till completed or failed
        data_dictionary = None
        while not self.learner_halt.is_set():
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
                self.connection_fifo.send('<RETRY>')
                failures += 1
                if failures > 3:
                    break

        return data_dictionary

    def _update_values(self):
        # let the spooler we're ready to update the values
        self.connection_fifo.send('<CA>')

        data_dictionary = self._read_json()
        if data_dictionary is not None:
            params = data_dictionary.get('parameters', '[[Dictionary Failure]]')
            costs = data_dictionary.get('costs', '[[Dictionary Failure]]')

            # update the parameters
            self._memory['costs'] = costs
            self._memory['parameters'] = params
            self.log.debug('Updated parameter and cost values.')
            self._state = LearnerState.TRAIN
            self.connection_fifo.send('<CA>')

            return True

        # bail otherwise
        self.log.debug('Failed to update parameter and cost values.')
        return False

    def _get_state(self):
        """
        Return the current state of the learner across the connection
        :return: None
        """
        self.connection_fifo.send(json.dumps(self._state))

    def _get_next_parameters(self):
        self.connection_fifo.send(json.dumps({'params': None}))

    def _get_config(self):
        self.connection_fifo.send('<CA>')
        data_dict = self._read_json()
        self.log.debug(f'Config received: {data_dict}')

        if data_dict is not None:
            self.connection_fifo.send('<CR>')
            for key, value in data_dict.items():
                self._memory[key] = value
            self.log.info('Updated configuration values.')
            self._state = LearnerState.CONFIGURED
        else:
            self.connection_fifo.send('<ERROR>')

    def train(self):
        pass

    def minimise(self):
        pass

    def ready(self):
        self._state = LearnerState.READY

    def halt(self):
        self.learner_halt.set()
        self.connection.close()


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def on_epoch_begin(self, epoch, logs=None):
        if not epoch % 10:
            self.log.info(f'Training epoch: {epoch}')


class NeuralNetLearner(BaseLearner):
    def __init__(self, learner_id):
        super().__init__(learner_id)

        # model for the optimisation
        self.model = None
        self.model_callbacks = [None, CustomCallback(self.log)]
        self.model_first_training = True
        self.scalers_trained = False
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self._memory['loss_history'] = []

        # worker thread for doing async work
        self.worker = None

        # next params queue
        self._next_params_queue = Queue()

    def initialise(self):
        super()._initialise()
        self.log.info(f'Starting learner [{self.id}] ...')

        self.log.info('Waiting to for configuration before starting model.')
        while not self.learner_halt.is_set():
            if self._state != LearnerState.CONFIGURED:
                sleep(0.1)
                continue
            else:
                break

        if self._state != LearnerState.CONFIGURED:
            self.log.error('Initialisation halted before configuration.')
            return False

        # create the model
        self.create_model(self._memory['n_inputs'])
        self._state = LearnerState.READY

    def create_model(self, inputs):
        # hyper params
        n_layers = 5
        neurons = 64

        # create the sequential layers
        input_layer = tf.keras.Input((inputs,))
        new_layer = input_layer
        for _ in range(n_layers):
            new_layer = tf.keras.layers.Dense(neurons,
                                              activation=tf.nn.swish,
                                              activity_regularizer=tf.keras.regularizers.L2(1e-8),
                                              kernel_initializer='he_normal',
                                              bias_initializer='ones'
                                              )(new_layer)

        # output layer
        output_layer = tf.keras.layers.Dense(1,
                                             activity_regularizer=tf.keras.regularizers.L2(1e-8))(new_layer)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        self.log.info(f'Created tf.keras model with layers {n_layers}x{neurons}. Inputs: {inputs}.')

    def train(self):
        self.worker = Thread(target=self.train_function)
        self.worker.start()

    def train_function(self):
        self._state = LearnerState.BUSY
        costs = np.array(self._memory['costs'])
        params = np.array(self._memory['parameters'])

        self.log.info(f'{costs.shape}, {params.shape}')

        # fix dimensions where necessary
        if len(costs.shape) < 2:
            costs = costs.reshape(-1, 1)
        if len(params.shape) < 2:
            params = params.reshape(-1, 1)

        # train the scalers if we need to
        if not self.scalers_trained:
            self.input_scaler.fit(params)
            self.output_scaler.fit(costs)
            self.scalers_trained = True

        self.log.info('Training model ...')

        if not self.model_first_training:
            self.model_callbacks[0] = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        else:
            self.model_callbacks[0] = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
            self.model_first_training = False

        # scale the inputs
        scaled_params = self.input_scaler.transform(params)
        scaled_costs = self.output_scaler.transform(costs)

        # fit the model
        history = self.model.fit(scaled_params, scaled_costs,
                                 epochs=self.config.get('model_epochs', 50),
                                 batch_size=32,
                                 callbacks=self.model_callbacks,
                                 verbose=0,
                                 shuffle=True)

        # record the loss value at the end of training
        self._memory['loss_history'].append(history.history['loss'][-1])
        self._state = LearnerState.MINIMISING

    def minimise(self):
        self._state = LearnerState.BUSY
        self.worker = Thread(target=self.minimise_function)
        self.worker.start()

    def minimise_function(self):
        # define a new parallel query
        # pq = ParallelQuery(self.model, bounds=self._memory['bounds'], log=self.log, input_scaler=self.input_scaler)
        self.log.info('Starting minimisation ... ')
        # results = pq.run_optim()
        # sorted_results = sorted(results, key=lambda X: X[1])
        # self.log.debug(f'New params: {sorted_results[0][0]}.')
        # self._next_params_queue.put(sorted_results[0][0])

        x0 = np.zeros(len(self._memory['bounds']))
        for idx, (bmin, bmax) in enumerate(self._memory['bounds']):
            x0[idx] = np.random.uniform(bmin, bmax, 1).squeeze()

        def min_func(X):
            X_scale = self.input_scaler.transform(X.reshape(1, -1))
            return self.model(X_scale)

        result = so.minimize(min_func, x0,
                             bounds=self._memory['bounds'],
                             options={'eps': 1e-3, })

        self.log.debug(f'New params: {result.x}.')
        self._next_params_queue.put(result.x)



        self._state = LearnerState.PREDICT

    def _get_next_parameters(self):
        next_params = self._next_params_queue.get()
        self._state = LearnerState.PREDICTED
        self.connection_fifo.send(json.dumps({'params': list(next_params)}))


class SamplingLearner(BaseLearner):
    def __init__(self, learner_id):
        super().__init__(learner_id)

        self.sample_points = None
        self.sample_iterator = 0

    def initialise(self):
        super()._initialise()

        self.log.info(f'Starting sampling learner [{self.id}] ...')

        self.log.info('Waiting to for configuration before generating points.')
        while not self.learner_halt.is_set():
            if self._state != LearnerState.CONFIGURED:
                sleep(0.1)
                continue
            else:
                break

        if self._state != LearnerState.CONFIGURED:
            self.log.error('Initialisation halted before configuration.')
            return False

        self.sample_points = self.sampling_grid(self._memory['initial_count'])
        self._state = LearnerState.READY

    def sampling_grid(self, initial_count=None):
        self.log.info('Sampling grid ...')

        bounds = self._memory.get('bounds', None)
        if bounds is None:
            raise RuntimeError('Bounds retrieved were none.')

        n_params = len(bounds)

        # local copy of the bounds
        upper_bounds = [b[1] for b in bounds]
        lower_bounds = [b[0] for b in bounds]

        # define the number of steps of reducing boundaries
        steps = 10

        # center and span of the boundaries, where we build the box from
        centers = []
        spans = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            centers.append((ub + lb) / 2.0)
            spans.append(abs(ub - lb))

        self.log.debug(f'Centers: {centers}, Spans:{spans}')

        # define how many points to grab per bound step
        if initial_count is None:
            points_per_iter = int(np.ceil((n_params * 2) / steps))
        else:
            points_per_iter = int(np.ceil(initial_count / steps))

        # account for user sillies
        if points_per_iter < 1:
            points_per_iter = 1

        self.log.debug(f'Steps:{steps}, PPI:{points_per_iter}, Bound len:{len(bounds)}.')

        # generate the step size for each bound
        step_sizes = []
        for spn in spans:
            step_sizes.append(spn / (2 * steps))
        self.log.debug(f'Step sizes: {step_sizes}')

        # for each step expand the bounds around the center and sample
        sample_points = [[] for x in range(len(centers))]
        for i in range(steps):
            new_bounds = []

            # find the new 'doughnut' bounds corresponding to (min_l, max_l) and (min_u, max_u)
            for idx, c in enumerate(centers):
                min_l = c - (step_sizes[idx] * i)
                max_l = c - (step_sizes[idx] * (i + 1))

                min_u = c + (step_sizes[idx] * i)
                max_u = c + (step_sizes[idx] * (i + 1))

                new_bounds.append([min_l, max_l, min_u, max_u])

            # for the new bounds switch randomly between the upper and lower bound
            for idx, bounds in enumerate(new_bounds):
                for itern in range(points_per_iter):
                    if np.random.rand() > 0.5:
                        point = np.random.uniform(bounds[2], bounds[3], 1)
                    else:
                        point = np.random.uniform(bounds[0], bounds[1], 1)

                    sample_points[idx].append(point)

        sample_points = np.array(sample_points).squeeze()
        self.log.debug(f'Generated samples with shape: {sample_points.shape}')

        if sample_points.shape[1] > initial_count:
            sample_points = sample_points[:, :initial_count]
            self.log.debug('Stripped samples to match initial_count')

        if sample_points.shape[1] < initial_count:
            self.log.warning('Samples generated was less than the initial count.')

        self.log.info(f'Generated sample points with shape: {sample_points.T.shape}')
        return sample_points.T.tolist()

    def _get_next_parameters(self):
        self.connection_fifo.send(json.dumps({'params': self.sample_points[self.sample_iterator]}))
        self.sample_iterator += 1


class ParallelQuery:
    def __init__(self, model, bounds, log, input_scaler):
        self.model = model
        self.N_threads = 5
        self.pool = ThreadPool(self.N_threads)
        # pairs of pipes (host, client)
        self.pipes = [Pipe() for _ in range(self.N_threads)]
        self.done = [Event() for _ in range(self.N_threads)]
        self.bounds = bounds
        self.evaluation_thread = Thread(target=self.min_function)
        self.evaluation_halt = Event()
        self.log = log
        self.input_scaler = input_scaler

    def comm_func(self, X, conn):
        conn.send(X)
        value = conn.recv()
        return value

    def target_func(self, id):
        _, conn = self.pipes[id]
        x0 = np.zeros(len(self.bounds))
        for idx, (bmin, bmax) in enumerate(self.bounds):
            x0[idx] = np.random.uniform(bmin, bmax, 1).squeeze()
        result = so.minimize(lambda X: self.comm_func(X, conn), x0,
                             bounds=self.bounds,
                             options={'eps': 1e-3,},
                             callback=lambda z: z)
        print(result)
        self.done[id].set()
        while True:
            conn.send(0)
            val = conn.recv()
            if val is None:
                break

        return result.x, result.fun

    def min_function(self):
        iter = 0
        while not self.evaluation_halt.is_set():
            values = np.zeros(self.N_threads*len(self.bounds)).reshape(self.N_threads, len(self.bounds))
            for idx, (conn, _) in enumerate(self.pipes):
                X = conn.recv()
                values[idx, :] = np.array(X)

            values = self.input_scaler.transform(values)

            preds = self.model.predict(values)
            iter += 1
            for idx, (conn, _) in enumerate(self.pipes):
                conn.send(preds[idx])

            counter = 0
            for e in self.done:
                if e.is_set():
                    counter += 1
            if counter >= self.N_threads:
                break

        for idx, (conn, _) in enumerate(self.pipes):
            conn.send(None)

    def run_optim(self):
        self.evaluation_thread.start()
        result = self.pool.map(self.target_func, range(self.N_threads))
        self.evaluation_halt.set()
        self.evaluation_thread.join()
        return result