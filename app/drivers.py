"""
Main driver class that specifies how to interact with the various equipments.
Author: Tranter Tech
Date: 2024
"""
import logging
import session
from utils.tools import load_config
from enum import Enum
from threading import Thread, Event
from queue import Queue, Full, Empty
from time import sleep
from PIL import Image
from .display import ok_clr, fail_clr

# get the logs
_LOG = logging.getLogger('Drivers')
# if session.display_log_handler is not None:
#     _LOG.addHandler(session.display_log_handler)
# else:
#     logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')

# supress numba as it is too noisey
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# load the config and devices
_CONFIG = load_config(_LOG)
_DEVICE_LIST = [_.lower() for _ in _CONFIG.get('devices', [])]
_DEVICE_FAILURES = ['' for _ in _DEVICE_LIST]
_DEVICE_KEYS = _CONFIG.get('device_keys', None)

# load in the devices we need to use
for device_str in _DEVICE_LIST:
    try:
        # load the respective drivers
        if device_str == 'thorlabs':
            from pylablib.devices import Thorlabs
        elif device_str == 'zaber':
            import zaber_motion as zm
            import zaber_motion.ascii as zma
        elif device_str == 'xeryon':
            from app.driver_libs import Xeryon
        elif device_str == 'jena':
            from app.driver_libs.jena import NV40
        elif device_str == 'tc038':
            from instruments.hcp import TC038
            from instruments.util_fns import split_unit_str
        elif device_str == 'thorlabs_camera':
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

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
    """
    Base class for the drivers used in the experimental setups. It is expected that many of the methods will be
    overriden with this class mostly serving as a template for future drivers.
    """
    def __init__(self):
        self._current_state = []
        self._error_queue = Queue()

    def set_parameters(self, X, asynch=True):
        """
        Given a set of parameters (generally 1-dimensional), facilitate the implementation of that parameter through
        the driver.
        :param X: The parameter to be implemented in the natural coordinate space of the driver.
        :param asynch: Boolean for whether this option should be performed asynchronously or not.
        :return: None
        """
        pass

    def get_current_parameters(self):
        """
        Returns the current state associated with the device driven by this driver.
        :return: Current state defined by the driver.
        """
        return self._current_state

    def add_error(self, error_message, additional_info=None):
        """
        Queue an error to be consumed later.
        :param error_message: The message associated with the error.
        :param additional_info: Any information that is to be passed with the error.
        :return: None
        """
        self._error_queue.put((error_message, additional_info))

    def shutdown(self):
        """
        Any clean up operations that need to be performed on shutdown. This should be called on exit.
        :return: None
        """
        pass


class ActuatorAction(Enum):
    """
    Enum for the actions an actuator can undertake. Currently supported are HOME, MOVE and STOP.
    """
    MOVE = 0
    HOME = 1
    STOP = 2


class ActuatorState(Enum):
    """
    Enum for the current state of the actuator. Currently supported are READY and MOVING.
    """
    READY = 0
    MOVING = 1


class KDC101(BaseDriver):
    """
    Controller class for the K-Cubes KDC101 supplied by Thorlabs. The following is implemented using pylablib and
    following documentation @ https://pylablib.readthedocs.io/en/latest/devices/Thorlabs_kinesis.html
    """
    def __init__(self, device_conn, actuator_id='', start_at_home=False, tol=0.0):
        """
        Constructor for the K-Cube driver which establishes a connection and intialises the actuators.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        :param start_at_home: Whether the actuator should home initially or not
        """
        super().__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._actuator = Thorlabs.KinesisMotor(device_conn, scale='stage')
        self._actuator_halt = Event()
        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.log.debug(f'Stage using units: {self._actuator.get_scale_units()}')
        self.async_running = Event()
        self._current_state = 0
        self._tol = tol

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
        self._current_state = self._actuator.get_position()

    def get_position(self, query=True):
        """
        Return the position of the actuator according to the device and driver.
        :param query: Determines whether the device should be queried.
        :return: Current state of the actuator.
        """
        if query:
            self._current_state = self._actuator.get_position()
        return self._current_state

    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY and not self.async_running.is_set() and not self._actuator.is_moving():
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
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('HOME', None))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('MOVE', arg),))
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
            self._actuator.move_to(float(value), scale=True)
            observed_twice = False
            while not (float(value) - self._tol <= self._actuator.get_position() <= float(value) + self._tol) and not observed_twice:
                sleep(0.1)
                if float(value) - self._tol <= self._actuator.get_position() <= float(value) + self._tol:
                    observed_twice = True
            self._state = ActuatorState.READY
            self._current_state = self._actuator.get_position()
            self.log.debug(f'Moved actuator to: {self._current_state}')

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        self._actuator.close()


class K10CR(BaseDriver):
    """
    Controller class for the K-Cubes KDC101 supplied by Thorlabs. The following is implemented using pylablib and
    following documentation @ https://pylablib.readthedocs.io/en/latest/devices/Thorlabs_kinesis.html
    """
    def __init__(self, device_conn, actuator_id='', start_at_home=False, tol=0.0):
        """
        Constructor for the K-Cube driver which establishes a connection and intialises the actuators.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        :param start_at_home: Whether the actuator should home initially or not
        """
        super().__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._actuator = Thorlabs.KinesisMotor(device_conn, scale='stage')
        self._actuator_halt = Event()
        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.log.debug(f'Stage using units: {self._actuator.get_scale_units()}')
        self.async_running = Event()
        self._current_state = 0
        self._tol = tol

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
        self._current_state = self._actuator.get_position()

    def get_position(self, query=True):
        """
        Return the position of the actuator according to the device and driver.
        :param query: Determines whether the device should be queried.
        :return: Current state of the actuator.
        """
        if query:
            self._current_state = self._actuator.get_position()
        return self._current_state

    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY and not self.async_running.is_set() and not self._actuator.is_moving():
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
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('HOME', None))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('MOVE', arg),))
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
            self._actuator.move_to(float(value), scale=True)
            observed_twice = False
            while not (float(value) - self._tol <= self._actuator.get_position() <= float(value) + self._tol) and not observed_twice:
                sleep(0.1)
                if float(value) - self._tol <= self._actuator.get_position() <= float(value) + self._tol:
                    observed_twice = True
            self._state = ActuatorState.READY
            self._current_state = self._actuator.get_position()
            self.log.debug(f'Moved actuator to: {self._current_state}')

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        self._actuator.close()


class KIM101(BaseDriver):
    """
    Controller class for the K-Cubes KDC101 supplied by Thorlabs. The following is implemented using pylablib and
    following documentation @ https://pylablib.readthedocs.io/en/latest/devices/Thorlabs_kinesis.html
    """
    def __init__(self, device_conn, actuator_id='', start_at_home=False, tol=0):
        """
        Constructor for the K-Cube driver which establishes a connection and intialises the actuators.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        :param start_at_home: Whether the actuator should home initially or not
        """
        super().__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._actuator = Thorlabs.KinesisPiezoMotor(device_conn)
        self._actuator_halt = Event()
        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.async_running = Event()
        self._tol = tol

    def initialise(self, config_dict):
        """
        Initialisation of the actuators that can be called by the interface/user
        :param config_dict: Configuration dictionary for initialising
        :return: None
        """
        if config_dict.get('home', False):
            self.log.info('Beginning homing sequence.')
            self.asynch_action(ActuatorAction.HOME)

        channels = config_dict.get('channels', (1, 2))
        self._actuator.enable_channels(channels)

        # reset the current state
        self._current_state = self.get_position()

    def get_position(self, query=True):
        """
        Return the position of the actuator according to the device and driver.
        :param query: Determines whether the device should be queried.
        :return: Current state of the actuator.
        """
        if query:
            channels = self._actuator.get_enabled_channels()
            self._current_state = [self._actuator.get_position(channel=c) for c in channels]
        return self._current_state

    def wait_on_actuator(self, axis=1):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY and not self.async_running.is_set() and not self._actuator.is_moving(channel=int(axis)):
                return True

            sleep(0.01)     # rate limit

        return False

    def set_parameters(self, X, asynch=False, axis=1):
        """
        Set parameters function overriding the base class implementation. This is how we expect to move the actuators in
        the interface.
        :param X: Parameters, in this case a scalar position
        :param asynch: whether we wish to perform this action asynchronously
        :return: None
        """
        self.asynch_action(ActuatorAction.MOVE, (X, axis))
        if not asynch:
            self.wait_on_actuator(axis=axis)

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
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('HOME', None))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('MOVE', arg),))
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
            position, axis = value
            axis = int(axis)
            self._actuator.move_to(int(position), channel=axis)
            while self._actuator.is_moving(channel=int(axis)):
                sleep(0.01)
            # observed_twice = False
            # while not (int(position) - self._tol <= self._actuator.get_position(channel=axis) <= int(position) + self._tol) and not observed_twice:
            #     sleep(0.1)
            #     if int(position) - self._tol <= self._actuator.get_position(channel=axis) <= int(position) + self._tol:
            #         observed_twice = True
            self._state = ActuatorState.READY
            self._current_state[axis-1] = self._actuator.get_position(channel=axis)
            self.log.debug(f'Moved actuator to: {self.get_position()}')

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        self._actuator.close()


class ZaberDriver(BaseDriver):
    """
    Class for controlling the Zaber X-VSR20A Stage (item 6)
    https://software.zaber.com/motion-library/docs/tutorials/install/py
    """

    def __init__(self, device_conn, actuator_id=''):
        """
        Constructor for the Zaber driver which establishes a connection and intialises the actuator.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        """
        super().__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._connection = zma.Connection.open_serial_port(device_conn)
        self._device = None
        self._actuator = None
        self._actuator_halt = Event()
        self._current_state = 0

        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.async_running = Event()

        # default units to use
        self.units = zm.Units.LENGTH_MILLIMETRES

    def initialise(self, config_dict={}):
        """
        Initialisation of the actuators that can be called by the interface/user
        :return:
        """
        

        # find an axis for us to initialise
        dev_ls = self._connection.detect_devices()
        try:
            self._device = dev_ls[0]
        except IndexError as e:
            self.log.error('Could not find Zaber device. Check connection.')
            self.log.error(f'{e.args}')
            return False
        self._actuator = self._device.get_axis(1)

        # reset the current state
        self._current_state = self.get_position()

        if config_dict.get('home', False):
            self.log.info('Beginning homing sequence.')
            self.asynch_action(ActuatorAction.HOME)

    def get_position(self, query=True):
        """
        Return the position of the actuator according to the device and driver.
        :param query: Determines whether the device should be queried.
        :return: Current state of the actuator.
        """
        if query:
            self._current_state = self._actuator.get_position(self.units)
        return self._current_state
    
    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY and not self.async_running.is_set():
                return True

            sleep(0.01)     # rate limit

        return False

    def set_parameters(self, X, asynch=False):
        """
        Set parameters function overriding the base class implementation. This is how we expect to move the actuators in
        the interface.
        :param X: Parameters, in this case a scalar position in self.units
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
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('HOME', None),))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('MOVE', arg),))
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
            self._actuator.home()   # this should be blocking
            self.log.debug('Homing complete.')
            self._state = ActuatorState.READY
        elif action == 'MOVE':
            # move to 'value' position in units
            self._actuator.move_absolute(value, self.units)
            self._state = ActuatorState.READY
            self._current_state = self._actuator.get_position(self.units)

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        self._connection.close()


class JenaDriver(BaseDriver):
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
        super().__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._actuator = NV40(device_conn)
        self._actuator_halt = Event()
        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.async_running = Event()

    def initialise(self, config_dict):
        """
        Initialisation of the actuators that can be called by the interface/user
        :param config_dict: Configuration dictionary for initialising
        :return:
        """
        # reset the current state
        self._current_state = self.get_position()

    def get_position(self, query=True):
        """
        Return the position of the actuator according to the device and driver.
        :param query: Determines whether the device should be queried.
        :return: Current state of the actuator.
        """
        if query:
            self._current_state = self._actuator.get_position()
        return self._current_state
    
    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
        """
        while not self._actuator_halt.is_set():
            # check if we are moving
            if self._state is ActuatorState.READY and not self.async_running.is_set():
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
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.HOME:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=('HOME', None))
            self._action_thread.start()
        elif action == ActuatorAction.MOVE:
            self._state = ActuatorState.MOVING
            self._action_thread = Thread(target=self._actions, args=(('MOVE', arg),))
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
            self._actuator.set_position(float(value))
            self._state = ActuatorState.READY
            self._current_state = self._actuator.get_position()
            self.log.debug(f'Moved actuator to: {self._current_state}')

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        # TODO: Confirm shutdown proc
        self._actuator.set_remote_control(False)


class XeryonDriver(BaseDriver):
    """
    Class for controlling the Xeryon Precision Model XLS-1 (item 5)
    https://xeryon.com/software/
    Note: The controller manual specifies running in closed loop, so we assume that is the case
    for this application.
    """
    def __init__(self, device_conn, actuator_id='', axis="X"):
        """
        Constructor for the Xeryon driver which establishes a connection and intialises the actuators.
        :param device_conn: connection string of the form (likely) /dev/TTY0...
        :param actuator_id: an ID string for keeping track of multiple actuators
        :param axis: The axis to be added to this particular actuator, should be X or Y
        """
        super(BaseDriver).__init__()
        self.ID = actuator_id
        self._action_thread = None
        self._state = ActuatorState.READY
        self._controller = Xeryon.Xeryon(device_conn, 115200)
        self._actuator = self._controller.addAxis(Xeryon.Stage.XLS_1, axis)
        self._actuator_halt = Event()

        self.log = logging.getLogger(f'Actuator:{actuator_id}')
        self.log.addHandler(session.display_log_handler)
        self.async_running = Event()

        self.units = Xeryon.Units.mu
        self.speed = 10     # this is in self.units/s.
        self.verbose = True

    def initialise(self):
        """
        Initialises the actuator and sets the relevant units and speed of the actuator.
        :return: None
        """
        # start the controller
        self._controller.start()

        # run the closed feedback loop to get the actuator to figure out where it is
        self._actuator.findIndex()

        # set the units and speed
        self._actuator.setUnits(self.units)
        self._actuator.setSpeed(self.speed)

    def wait_on_actuator(self):
        """
        Wait for the actuator to finish moving by blocking until it has.
        :return: Boolean indicating whether we had to wait at all.
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

        if self.async_running.is_set():
            raise RuntimeError(f'Cannot start a new action before the old action has completed. Check '
                               f'{self.__class__.__name__}.async_running.is_set() before calling a new action.')

        if action == ActuatorAction.MOVE:
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

        if action == 'MOVE':
            # move to 'value' steps
            success = self._actuator.setDPOS(value)
            if not success:
                self.add_error('Failed to move actuator ')
            self._state = ActuatorState.READY
            self._current_state = self._actuator.getDPOS()

        # clear the running command
        self.async_running.clear()

    def shutdown(self):
        # shutdown the controller
        self._controller.stop()
        

class TC038Driver(BaseDriver):
    """
    Driver class for the TC038 Temperature controller / crystal oven (item 8).
    https://github.com/instrumentkit/InstrumentKit/blob/main/src/instruments/hcp/tc038.py
    """
    def __init__(self, address, poll_delay=3.0, observations_required=4):
        """
        Driver for the TC038 which sets up a polling delay and set of observations to determine whether the temperature
        has satbilised.
        :param address: Address of the temperature controller.
        :param poll_delay: Delay between polling to allow stabilisation.
        :param observations_required: Number of observations of a stable temperature required to determine that the
        temperature is stable.
        """
        super().__init__()

        self.tc = TC038.open_serial(address)
        self.bounds = (50.0, 100.0)
        self.tol = 0.0
        self.poll_delay = poll_delay
        self.observations_required = observations_required

        self.log = logging.getLogger(f'Controller:TC038')
        self.log.addHandler(session.display_log_handler)

    def change_temp(self, temp, asynch=False):
        """
        Change the temperature set point.
        :param temp: Temperature as a float in degrees C
        :param asynch: Boolean to perform this action asynchronously or not.
        :return: None.
        """
        try:
            new_temp = float(temp)
            # clip the temperature
            new_temp = max(min(new_temp, self.bounds[1]), self.bounds[0])

            if new_temp != float(temp):
                self.log.warning(f'Temperature was coerced to bounds: {self.bounds}')

            if split_unit_str(self.tc.setpoint)[0] == new_temp:
                self.log.info('Value already set.')
                return

            self.tc.setpoint = new_temp
            if not asynch:
                self._monitor_temp()
        except ValueError:
            self.log.error(f'Error converting temperature {temp} to a float.')

    def measure_temp(self):
        """
        Measure the temperature according to the controller
        :return: Temperature in units of degrees C
        """
        # get the temperature
        return split_unit_str(self.tc.temperature)[0]
    
    def _monitor_temp(self):
        """
        Monitor whether the target temperature has been reached or not.
        :return: None
        """
        temp_reached = False
        observed = 0
        while not temp_reached:
            current_temp = self.measure_temp()
            set_temp = split_unit_str(self.tc.setpoint)[0]
            self.log.debug(f'Current T:{current_temp} - Set T:{set_temp} - obs:{observed}')
            # print(f'Current T:{current_temp} - Set T:{set_temp} - obs:{observed}')

            if abs(current_temp - set_temp) <= self.tol:
                observed += 1

            if observed > self.observations_required:
                temp_reached = True
            else:
                # sleep until the next measurement
                sleep(self.poll_delay)

        self.log.info("Target temperature reached.")

        


class CameraDriver(BaseDriver):
    """
    Driver for the Thorlabs camera, used in the quantum imagin experiment as the main measurement.
    """
    def __init__(self, camera_settings=None, image_settings=None):
        """
        Constructor for the camera driver which brings in a number of camera settings and image settings.
        :param camera_settings: Camera settings, currently supports the exposure time setting in microseconds.
        :param image_settings: Image settings used in processing. Current supports setting an ROI, gain and bit_shift
        """
        super().__init__()
        self.log = logging.getLogger('Camera')
        self.camera = None
        self.app = None
        self.acquisition_thread = Thread(target=self._acquisition_loop)
        self._halt_acq = Event()
        self._acquire_img = Event()
        self._image_queue = Queue(maxsize=1)
        self.sdk = None

        if camera_settings is None:
            self.log.warning('No camera settings provided. Using defaults.')
            self.camera_settings = {}
        else:
            self.camera_settings = camera_settings

        if image_settings is None:
            image_settings = {}

        # image processing values
        self.roi = image_settings.get('roi', (0, 1000, 500, 1500))
        self.gain = image_settings.get('gain', 3)
        self.bit_shift = image_settings.get('bit_shift', 12)
        self._bit_depth = None
        self.return_as_numpy_array = False

    def initialise(self):
        """
        Initialise the camera. This must be called at least once per instantiation of the camera driver to ensure
        communication occurs.
        :return: None
        """
        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()

        if len(camera_list) < 1:
            self.log.error('Could not find the camera ...')
            return False

        # load the camera
        self.camera = self.sdk.open_camera(camera_list[0])

        # set up the camera
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.exposure_time_us = self.camera_settings.get('exposure_time', 100000)
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        self._bit_depth = self.camera.bit_depth
        self.camera.image_poll_timeout_ms = 0

    def start_acquisition(self):
        """
        Main entry point for starting the acquisition thread.
        :return: None
        """
        # run the acquistion thread
        self.acquisition_thread.start()

    def _get_image(self, frame):
        """
        Process a frame from the camera and return an image. Adapted from the Thorlabs example and Thomas' code.
        :param frame: Frame object as passed from the Thorlabs camera driver.
        :return: Image as an array.
        """
        if not(self.return_as_numpy_array):
            # no coloring, just scale down image to 8 bpp and place into PIL Image object
            scaled_image = (frame.image_buffer >> (self._bit_depth - self.bit_shift)) * self.gain
            xmin, xmax, ymin, ymax = self.roi
            scale = 0.25
            new_x = int((xmax - xmin) * scale)
            new_y = int((ymax - ymin) * scale)
            img = Image.fromarray(scaled_image[xmin:xmax, ymin:ymax])
            try:
                if scale != 1.0:
                    img = img.resize((new_x, new_y))
            except Exception as e:
                print(e.args)

            return img
        else:
            scaled_image = frame.image_buffer
            return np.copy(scaled_image)

    def _acquisition_loop(self):
        """
        Main acquisition loop in charge of getting frames from the camera adding them to the queue.
        :return: None
        """
        while not self._halt_acq.is_set():
            try:
                frame = self.camera.get_pending_frame_or_null()
                if frame is not None and self._acquire_img.is_set():
                    pil_or_numpy_array_image = self._get_image(frame)
                    self._acquire_img.clear()
                    self._image_queue.put_nowait(pil_or_numpy_array_image)
            except Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                self.log.error("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        self.log.info('Image acquisition halted.')

    def get_image(self, timeout=None):
        """
        Return an image when asked. Frames are discarded normally by the camera until this method is called.
        :param timeout: How long to wait in seconds when asking for a frame. None by default blocks until a frame is
        returned.
        :return: An image array.
        """
        # flag that we want an image
        self._acquire_img.set()

        # wait to get an image back
        image = self._image_queue.get(timeout=timeout)
        return image

    def shutdown(self):
        self._halt_acq.set()
        self.acquisition_thread.join()
        self.camera.dispose()
        self.sdk.dispose()


def main():
    pass
    # time to define some tests
    # _LOG.info('Starting a device ... ')
    # stage_0 = KDC101('/dev/ttyUSB5', 'test')
    # stage_0 = ZaberDriver('/dev/ttyUSB0', 'test')
    
    # while True:
    #     x = input('>')
    #     if x == 'q':
    #         break
    #     elif x == 'e':
    #         for i in range(stage_0._error_queue.qsize()):
    #             _LOG.error(stage_0._error_queue.get())
    #     elif x == 'init':
    #         stage_0.initialise()
    #     else:
    #         try:
    #             x_pos = float(x)
    #             stage_0.set_parameters(x_pos)
    #         except ValueError as e:
    #             _LOG.error(f'Could not convert {x} to float.')
    
    # stage_0.shutdown()
