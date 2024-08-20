"""

Author: Tranter Tech
Date: 2024
"""
import logging
import session
from utils.Tools import load_config
from enum import Enum
from threading import Thread, Event
from queue import Queue
from time import sleep

# get the logs
_LOG = logging.getLogger('Drivers')
_LOG.addHandler(session.display_log_handler)
# supress numba as it is too noisey
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# load the config and devices
_CONFIG = load_config(_LOG)
_DEVICE_LIST = [_.lower() for _ in _CONFIG.get('devices', [])]
_DEVICE_FAILURES = ['' for _ in _DEVICE_LIST]

ok_clr = lambda ret_str: '\033[32m' + ret_str + '\033[0m'
fail_clr = lambda ret_str: '\033[93m' + ret_str + '\033[0m'

# load in the devices we need to use
for device_str in _DEVICE_LIST:
    try:
        # load the respective drivers
        if device_str == 'thorlabs':
            from pylablib.devices import Thorlabs
        elif device_str == 'zaber':
            import zaber_motion as zm
        elif device_str == 'xeryon':
            from app.driver_libs import Xeryon
        elif device_str == 'jena':
            from app.driver_libs.jena import NV40
        elif device_str == 'tc038':
            from pymeasure.instruments.hcp import TC038

        _LOG.debug(f'Loading device drivers for {device_str}:' + ok_clr('\t[OK]'))
        _DEVICE_FAILURES[_DEVICE_LIST.index(device_str)] = f'{device_str.upper()}:OK'
    except Exception as e:
        _DEVICE_FAILURES[_DEVICE_LIST.index(device_str)] = f'{device_str.upper()}: {e.args}'
        _LOG.debug(f'Loading device drivers for {device_str}:' + fail_clr('\t[FAIL]'))

# log the stats about what devices were loaded
_LOG.info('Device stats: -- ')
for device in _DEVICE_FAILURES:
    _LOG.info('\t' + device)


class BaseDriver:
    def __init__(self):
        self._current_state = []
        self._error_queue = Queue()

    def set_parameters(self, X, asynch=True):
        pass

    def get_current_parameters(self):
        return self._current_state

    def add_error(self, error_message, additional_info=None):
        self._error_queue.put((error_message, additional_info))


class ActuatorAction(Enum):
    """
    Enum for the actions an actuator can undertake
    """
    MOVE = 0
    HOME = 1
    STOP = 2


class ActuatorState(Enum):
    """
    Enum for the current state of the actuator
    """
    READY = 0
    MOVING = 1


class KDC101(BaseDriver):
    """
    Controller class for the K-Cubes KDC101 supplied by Thorlabs. The following is implemented using pylablib and
    following documentation @ https://pylablib.readthedocs.io/en/latest/devices/Thorlabs_kinesis.html
    """
    def __init__(self, device_conn, actuator_id='', start_at_home=False):
        """
        Constructor for the K-Cube driver which establishes a connection and intialises the actuators.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        :param start_at_home: Whether the actuator should home initially or not
        """
        super(BaseDriver).__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._actuator = Thorlabs.KinesisMotor(device_conn)
        self._actuator_halt = Event()
        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.async_running = Event()

    def initialise(self, config_dict):
        """
        Initialisation of the actuators that can be called by the interface/user
        :param config_dict: Configuration dictionary for initialising
        :return:
        """
        if config_dict.get('home', False):
            self.log.info('Beginning homing sequence.')
            self.asynch_action(ActuatorAction.HOME)

        # reset the current state
        self._current_state = [0]

    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving
        :return:
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY:
                return True

            sleep(0.01)     # rate limit

        return False

    def set_parameters(self, X, asynch=False):
        """
        Set parameters function overriding the base class implementation. This is how we expect to move the actuators in
        the interface.
        :param X: Parameters, in this case a scalar position
        :param asynch: whether we wish to perform this action asynchronously
        :return: None
        """
        self.asynch_action(ActuatorAction.MOVE, X)
        if not asynch:
            self.wait_on_actuator()

    def asynch_action(self, action=None, arg=None):
        """
        Asynchronous action to be taken by the actuators.
        :param action: The action to take, should parse ActuatorAction class.
        :param arg: Any argument that needs to be provided. For example ActuatorAction.MOVE requires a scalar distance.
        :return:Success (bool)
        """
        if action is None:
            raise RuntimeError('No action provided to the actuator.')

        # halt if necessary
        if action == ActuatorAction.STOP:
            self._actuator.stop()
            return True

        if self.async_running.is_set():
            raise RuntimeError('Cannot start a new action before the old action has completed. Check '
                               'KDC101.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('HOME', None))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('MOVE', arg))
            self._action_thread.start()

        return True

    def _actions(self, args):
        """
        Asynchronous thread to call, calls to respective functions are blocking.
        :param args: args parsed from the asynch_action function in the form (action, value)
        :return: None
        """
        # get the arguments
        action, value = args
        self.async_running.set()

        if action == 'HOME':
            self._actuator.home()
            self._actuator.wait_for_home()
            self._state = ActuatorState.READY
        elif action == 'MOVE':
            # move to 'value' steps
            success = self._actuator.move_to(int(value), scale=False)
            if not success:
                self.add_error('Failed to move actuator ')
            self._state = ActuatorState.READY
            self._current_state = self._actuator.get_position()

        # clear the running command
        self.async_running.clear()


class ZaberDriver(BaseDriver):
    """
    Class for controlling the Zaber X-VSR20A Stage (item 6)
    https://software.zaber.com/motion-library/docs/tutorials/install/py
    """
    def __init__(self):
        super().__init__()


class XeryonDriver(BaseDriver):
    """
    Class for controlling the Xeryon Precision Model XLS-1 (item 5)
    https://xeryon.com/software/
    """
    def __init__(self):
        super().__init__()
        

class TC038Driver(BaseDriver):
    """
    Driver class for the TC038 Temperature controller / crystal oven (item 8).
    https://pymeasure.readthedocs.io/en/latest/api/instruments/hcp/tc038.html
    """
    def __init__(self):
        super().__init__()