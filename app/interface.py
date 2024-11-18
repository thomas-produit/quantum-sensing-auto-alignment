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
        self.plot_queue = None

    def run_initialisation(self, args=None):
        return True

    def run_parameters(self, params, args=None):
        pass

    def set_plot(self, queue):
        self.plot_queue = queue


class TestInterface(BaseInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def run_parameters(self, params, args=''):
        self.plot_queue.put(('replace', {'y': [np.random.uniform(0, 255, (50, 50))]}, 'fringe_img'))

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


# import the relevant functions for processing
from utils.cost_tools import find_circular_FOV, create_circular_mask
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

        # signal arm, horizontal
        sig_arm_horz = KDC101('/dev/ttyUSB4', 'sig_arm_horz', tol=1e-4)
        sig_arm_horz.initialise(config_dict={})
        self._actuators.append(sig_arm_horz)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # signal arm, vertical
        sig_arm_vert = KDC101('/dev/ttyUSB6', 'sig_arm_vert', tol=1e-4)
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

        # grab the arguments and see if we need to run a parameter
        if args is not None:
            init_params = args.get('init_params', None)

            if init_params is not None:
                self.run_parameters(init_params, args={'skip_data': True})

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

        if args is None:
            args = {}

        skip_data = bool(args.get('skip_data', False))
        get_scale = bool(args.get('get_scale', False))
        scale = args.get('scale', (1, 1))

        # ------------------------------
        # --- Data Acquisition
        # ------------------------------
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

            self.actuator_list['idler_shutter'].set_parameters(0, asynch=True)

            grp = self.save_file.create_group(f'run_{self._counter}')
            grp.create_dataset('dark_img', data=dark_img)
            grp.create_dataset('fringes', data=img_arr)
            grp.create_dataset('params', data=params)

            # plot the various images
            # TODO: Not hotfixes please
            self.plot_queue.put(('replace', {'y': np.clip(img_arr[0], 0, 800)}, 'fringe_img'))
            self.plot_queue.put(('replace', {'y': np.clip(dark_img, 0, 800)}, 'dark_img'))

            self._counter += 1

            return self._cost(img_arr, dark_img, get_scale, scale)

        # if we skip data, return None
        return None

    def _cost(self, img_in, dark_img_in, get_scale=False, scale=(1, 1)):
        img = np.copy(img_in)
        dark_img = np.copy(dark_img_in)
        radius = 55
        center = (145, 125)

        # define the bounds
        edge_px = 5
        ymin = int(center[0] - radius - edge_px)
        ymax = int(center[0] + radius + edge_px)
        xmin = int(center[1] - radius - edge_px)
        xmax = int(center[1] + radius + edge_px)
        height, width = dark_img.shape

        # crop the images
        cropped_imgs = img[:, ymin:ymax, xmin:xmax]
        cropped_dark_img = dark_img[ymin:ymax, xmin:xmax]

        # create a mask
        mask = create_circular_mask(height, width, center=center, radius=radius * 1.0)
        cropped_mask = mask[ymin:ymax, xmin:xmax]

        # take a fft along the sample dimension (dim 0)
        which_axis_fft = 0
        fft = np.abs(np.fft.rfft(cropped_imgs, axis=which_axis_fft, norm='forward'))

        # get the DC component and Amplitude
        DC_fft = fft[0]  # Smart slicing
        Amplitude_fft = 2 * np.max(fft[1:], axis=which_axis_fft)

        # calculate visibility
        Visibility_raw = Amplitude_fft / DC_fft

        # Mask Visibility image to FOV
        Visibility = np.ma.masked_array(Visibility_raw, mask=cropped_mask)
        Visibility = np.ma.filled(Visibility, fill_value=0)

        ROI_mask_big = create_circular_mask(height, width, center=center, radius=radius)

        # camera settings
        Camera_bit_depth = 16
        Camera_max_value = 2 ** Camera_bit_depth - 1

        Dark_img_ROI = np.ma.masked_array(dark_img, mask=ROI_mask_big)

        # calculate cost components
        Cost_average = np.ma.sum(Dark_img_ROI) / (Dark_img_ROI.count() * Camera_max_value)
        Weight_average = 10**scale[0]

        Cost_visibility = np.ma.average(Visibility)
        Weight_visibility = 10**scale[1]

        cost = - (Weight_average * Cost_average + Weight_visibility * Cost_visibility)

        if get_scale:
            return Cost_average, Cost_visibility
        else:
            return cost

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