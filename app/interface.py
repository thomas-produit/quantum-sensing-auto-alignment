"""

Author: Tranter Tech
Date: 2024
"""
import numpy as np
from app.drivers import KDC101, KIM101, K10CR, JenaDriver, ZaberDriver, CameraDriver
import logging
import session
from time import sleep
import h5py
from datetime import datetime

class BaseInterface:
    def __init__(self):
        pass

    def run_initialisation(self, args=None):
        return True

    def run_parameters(self, params, args=None):
        pass


class TestInterface(BaseInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def run_parameters(self, params, args=''):
        return self._cost_function(params, args)

    def _cost_function(self, params, args):
        params = np.array(params)
        if args == 'ackley':
            val = self.ackley_func(params)
        elif args == 'parabola':
            val = np.sum(np.square(params))
        else:
            raise RuntimeError(f'Parameter for function {args} does not match any function.')
        return val

    @staticmethod
    def ackley_func(chromosome):
        """"""
        firstSum = 0.0
        secondSum = 0.0
        for c in chromosome:
            firstSum += c ** 2.0
            secondSum += np.cos(2.0 * np.pi * c)
        n = float(len(chromosome))
        return -20.0 * np.exp(-0.2 * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + 20 + np.exp(1)


class QuantumImaging(BaseInterface):
    def __init__(self, save_dir=None):
        super().__init__()

        self._actuators = []
        self.actuator_list = {}
        self._camera = None
        self.log = logging.getLogger('QIInterface')
        self.log.addHandler(session.display_log_handler)

        # TODO: Error check this
        if save_dir is None:
            now_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
            save_dir = f'./data/{now_str}_data.h5'
        self.save_file = h5py.File(save_dir, 'w')

        self._counter = 0

    def run_initialisation(self, args=None):
        # Camera init
        self._camera = CameraDriver()
        self._camera.initialise()
        self._camera.start_acquisition()
        self.log.info('Starting Camera ...')

        # Longitudinal
        long_act = KDC101('/dev/ttyUSB8', 'longitudinal', tol=1e-4)
        long_act.initialise(config_dict={})
        self._actuators.append(long_act)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # lateral
        lat_act = KDC101('/dev/ttyUSB7', 'lateral')
        lat_act.initialise(config_dict={})
        self._actuators.append(lat_act)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # signal arm piezos
        # sig_arm_piezo = KIM101('/dev/ttyUSB3', 'sig_arm_piezo')
        # sig_arm_piezo.initialise(config_dict={})
        # self._actuators.append(sig_arm_piezo)
        # self.log.info(f'Starting {self._actuators[-1].ID} ...')

        sig_arm_horz = KDC101('/dev/ttyUSB6', 'sig_arm_horz', tol=1e-4)
        sig_arm_horz.initialise(config_dict={})
        self._actuators.append(sig_arm_horz)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        sig_arm_vert = KDC101('/dev/ttyUSB4', 'sig_arm_vert', tol=1e-4)
        sig_arm_vert.initialise(config_dict={})
        self._actuators.append(sig_arm_vert)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # idler shutter
        idler_shut = KDC101('/dev/ttyUSB5', 'idler_shutter', tol=1e-4)
        idler_shut.initialise(config_dict={})
        self._actuators.append(idler_shut)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # z coarse (Zaber)
        z_coarse = ZaberDriver('/dev/ttyUSB0', 'z_coarse')
        z_coarse.initialise(config_dict={})
        self._actuators.append(z_coarse)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # z fine (Jena)
        z_fine = JenaDriver('/dev/ttyUSB2', 'z_fine')
        z_fine.initialise(config_dict={})
        self._actuators.append(z_fine)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # HWP - intensity
        hwp = K10CR('/dev/ttyUSB1', 'HWP')
        hwp.initialise(config_dict={})
        self._actuators.append(hwp)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # # QWP - polarisation correction
        # qwp = K10CR('/dev/ttyUSB3', 'QWP')
        # qwp.initialise(config_dict={})
        # self._actuators.append(qwp)
        # self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # /USB3/ <- Temp control

        # compile them into a handy dictionary
        for actu in self._actuators:
            self.actuator_list[actu.ID] = actu

        return True

    def test_actuators(self):
        actuator = self.actuator_list['longitudinal']
        current_pos = actuator.get_position()
        actuator.set_parameters(current_pos + 100, asynch=False)
        actuator.set_parameters(current_pos - 100, asynch=False)

    def shutdown(self):
        try:
            for actu in self._actuators:
                self.log.info(f'Shutting down {actu.ID}.')
                if actu.ID == 'sig_arm_piezo':
                    actu.wait_on_actuator(axis=1)
                    actu.wait_on_actuator(axis=2)
                else:
                    actu.wait_on_actuator()
                actu.shutdown()
        except Exception as e:
            self.log.error(f'Shutdown function failed for actuator:{actu.id}.')
            self.log.error(e.args)

        self._camera.shutdown()
        self.save_file.close()

    def run_parameters(self, params, args=None):
        # ------------------------------
        # --- Parameter Setting
        # ------------------------------
        # param 0: longitdinal
        # param 1: Sig arm horizontal
        # param 2: Sig arm vertical
        # param 3: z coarse
        
        self.actuator_list['longitudinal'].set_parameters(params[0], asynch=True)
        self.actuator_list['z_coarse'].set_parameters(params[3], asynch=True)
        self.actuator_list['sig_arm_horz'].set_parameters(params[1], asynch=True)
        self.actuator_list['sig_arm_vert'].set_parameters(params[2], asynch=True)

        self.actuator_list['longitudinal'].wait_on_actuator()
        self.actuator_list['z_coarse'].wait_on_actuator()

        # ------------------------------
        # --- Data Acquisition
        # ------------------------------

        skip_data = False
        if args is not None:
            skip_data = bool(args)

        if not skip_data:
            # wait for the shutter to finish, if it is still moving
            while self.actuator_list['idler_shutter'].async_running.is_set():
                sleep(0.01)

            # open the shutter
            self.actuator_list['idler_shutter'].set_parameters(0)
            self.log.debug('Opened the shutter.')

            img_list = []
            _steps = 20
            for step in np.linspace(0, 3, _steps):
                self.actuator_list['z_fine'].set_parameters(step)
                new_img = np.array(self._camera.get_image(), dtype=np.float32)
                img_list.append(new_img)
            
            img_arr = np.array(img_list)
            self.log.debug(f'Acquired {_steps} steps and returned {img_arr.shape} array.')

            # close the shutter (20mm = closed)
            self.actuator_list['idler_shutter'].set_parameters(20e-3)
            self.log.debug('Closed the shutter.')


            # take an image
            dark_img = np.array(self._camera.get_image(), dtype=np.float32)

            # diff_arr = img_arr - dark_img
            # _threshold = 1
            # mask = diff_arr[diff_arr>]
            self.actuator_list['idler_shutter'].set_parameters(0, asynch=True)

            grp = self.save_file.create_group(f'run_{self._counter}')
            grp.create_dataset('dark_img', data=dark_img)
            grp.create_dataset('fringes', data=img_arr)
            grp.create_dataset('params', data=params)
            # fft_diff_arr = np.fft.rfft2(diff_arr, axis=0)

            self._counter += 1

        return np.random.rand()
        

def test():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')
    
    qii = QuantumImaging()
    qii.run_initialisation()
    # qii.test_actuators()

    while True:
        x = input('>')
        if x == 'q':
            break
        else:
            args = x.split(':')
            if len(args) < 2:
                qii.log.error(f'Not enough arguments provided in {x}')
                continue

            actuator = qii.actuator_list.get(args[0], None)

            if actuator is None:
                qii.log.error(f'Actuator with key: {args[0]} not found.')
                continue

            if args[1] == 'query':
                qii.log.info(actuator.get_position())
            else:
                try:
                    value = float(args[1])
                except ValueError:
                    qii.log.info(f'Failed to convert {args[1]} to float.')

                if len(args) == 2:
                    actuator.set_parameters(value)
                else:
                    actuator.set_parameters(value, False, int(args[2]))
            

    qii.shutdown()