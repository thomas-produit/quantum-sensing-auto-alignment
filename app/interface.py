"""
Main entry point for the interfaces that deal with the experiment. Logic for interacting with physical hardware outside
of driver implementations should be located here. An interface should be defined on an experiment by experiment basis
allowing the flexibility of the ML optimiser to be applied with instantiation of that particular interface.
Author: Tranter Tech
Date: 2024
"""

import numpy as np
from app.drivers import KDC101, KIM101, K10CR, JenaDriver, ZaberDriver, CameraDriver, _DEVICE_KEYS
import logging
import session
from time import sleep
import h5py
from datetime import datetime
from subprocess import run as cmd_run


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
        """
        Test function used for benchmarking the optimiser.
        :param chromosome: List of parameters to be fed to the Ackley function.
        :return: The cost associated with the supplied parameters.
        """
        firstSum = 0.0
        secondSum = 0.0
        for c in chromosome:
            firstSum += c ** 2.0
            secondSum += np.cos(2.0 * np.pi * c)
        n = float(len(chromosome))
        return -20.0 * np.exp(-0.2 * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + 20 + np.exp(1)


# import the relevant functions for processing
from utils.cost_tools import create_circular_mask, compute_zernike_decomposition, cost_evaluation, RZern
class QuantumImaging(BaseInterface):
    def __init__(self, save_dir=None):
        super().__init__()

        self._actuators = []
        self.actuator_list = {}
        self._camera = None
        self.log = logging.getLogger('QIInterface')
        self.log.addHandler(session.display_log_handler)

        # Attempt to open the save file
        try:
            if save_dir is None:
                now_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
                save_dir = f'./data/{now_str}_data.h5'
            self.save_file = h5py.File(save_dir, 'w')
        except OSError as e:
            error_msg = f'Could not open the data file \'{save_dir}\': {e.args}.'
            self.log.error(error_msg)
            raise RuntimeError(error_msg)

        self._counter = 0
        self.dark_img = None
        self.zernike_object = None

    def run_initialisation(self, args=None):
        """
        Runs the interface initialisation. This is called by the manager during start up with relevant arguments passed
        by the manager. If this method fails the interface will not be initialised and the optimisation will halt.
        :param args: Arguments that may be passed by the manager from the users initial configuration of the
        optimisation. There is no specified form as the user is responsible for these.
        :return: Boolean denoting the success of the interface initialisation.
        """
        device_dict, missing_devices = self._get_device_ports()
        if device_dict is None:
            self.log.error('Could not start interface as devices were not located.')
            return False

        # if there are specific actuators that should not be missing before startup, fail here
        deal_breaker_devs = ['longitudinal', 'sig_arm_horz', 'sig_arm_vert', 'z_coarse', 'z_fine']
        devices_missing = False
        for dev in deal_breaker_devs:
            if dev in missing_devices:
                self.log.error(f'Critical device [{dev}] missing.')
                devices_missing = True

        if devices_missing:
            self.log.error('Critical devices are missing, halting operation.')
            return False

        # Camera init
        self._camera = CameraDriver()
        self._camera.initialise()
        self._camera.start_acquisition()
        self.log.info('Starting Camera ...')

        # Longitudinal
        long_act = KDC101(device_dict['longitudinal'], 'longitudinal', tol=1e-4)
        long_act.initialise(config_dict={})
        self._actuators.append(long_act)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # lateral
        lat_act = KDC101(device_dict['lateral'], 'lateral')
        lat_act.initialise(config_dict={})
        self._actuators.append(lat_act)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # signal arm piezos
        # sig_arm_piezo = KIM101('/dev/ttyUSB3', 'sig_arm_piezo')
        # sig_arm_piezo.initialise(config_dict={})
        # self._actuators.append(sig_arm_piezo)
        # self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # signal arm, horizontal
        sig_arm_horz = KDC101(device_dict['sig_arm_horz'], 'sig_arm_horz', tol=1e-4)
        sig_arm_horz.initialise(config_dict={})
        self._actuators.append(sig_arm_horz)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # signal arm, vertical
        sig_arm_vert = KDC101(device_dict['sig_arm_vert'], 'sig_arm_vert', tol=1e-4)
        sig_arm_vert.initialise(config_dict={})
        self._actuators.append(sig_arm_vert)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # idler shutter
        idler_shut = KDC101(device_dict['idler_shutter'], 'idler_shutter', tol=1e-4)
        idler_shut.initialise(config_dict={})
        self._actuators.append(idler_shut)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # z coarse (Zaber)
        z_coarse = ZaberDriver(device_dict['z_coarse'], 'z_coarse')
        z_coarse.initialise(config_dict={})
        self._actuators.append(z_coarse)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # z fine (Jena)
        z_fine = JenaDriver(device_dict['z_fine'], 'z_fine')
        z_fine.initialise(config_dict={})
        self._actuators.append(z_fine)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # HWP - intensity
        hwp = K10CR(device_dict['HWP'], 'HWP')
        hwp.initialise(config_dict={})
        self._actuators.append(hwp)
        self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # # QWP - polarisation correction
        # qwp = K10CR('/dev/ttyUSB3', 'QWP')
        # qwp.initialise(config_dict={})
        # self._actuators.append(qwp)
        # self.log.info(f'Starting {self._actuators[-1].ID} ...')

        # compile them into a handy dictionary
        for actu in self._actuators:
            self.actuator_list[actu.ID] = actu

        # grab the arguments and see if we need to run a parameter
        if args is not None:
            init_params = args.get('init_params', None)

            # calculate the zernike methods
            if args.get('cost_definition', '') == 'zernike':
                self._start_zernike()

            # run the initial parameters if we have them
            if init_params is not None:
                self.run_parameters(init_params, args={'skip_data': False,
                                                       'use_shutter': True,
                                                       'fringe_steps': 0})

        return True

    def _get_device_ports(self):
        """
        Return the device ports for all the devices that are needed for the interface.
        :return: Tuple -> dictionary containing the link between the interface key and the serial key, None if it fails.
        Additionally returns a list of missing devices.
        """
        cmd = cmd_run(['ls', '-la', '/dev/serial/by-id/'], capture_output=True)
        missing_devices = []

        if cmd.stderr != b'':
            self.log.error(f"Could not run the device list command with the following error: "
                           f"{cmd.stderr.decode('UTF-8')}")
            return None, missing_devices

        device_dict = {}
        # --- CODE FOR IDENTIFYING THE DEVICES FROM THE STRINGS

        return device_dict, missing_devices


    def _start_zernike(self):
        # radial order that we will fit to
        radial_order = 10
        self.zernike_object = RZern(radial_order)

        # Hard coded
        L, K = 120, 120

        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)

        self.zernike_object.make_cart_grid(xv, yv)

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
        take_dark = bool(args.get('take_dark', False))
        scale = args.get('scale', (1, 1))
        use_shutter = bool(args.get('use_shutter', False))
        fringe_steps = args.get('fringe_steps', 20)
        cost_definition = args.get('cost_definition', '')

        # ------------------------------
        # --- Data Acquisition
        # ------------------------------
        if not skip_data:
            # wait for the shutter to finish, if it is still moving
            while self.actuator_list['idler_shutter'].async_running.is_set():
                sleep(0.01)

            # open the shutter
            if use_shutter:
                self.actuator_list['idler_shutter'].set_parameters(0)
                self.log.debug('Opened the shutter.')

            img_list = []
            _steps = fringe_steps
            for step in np.linspace(0, 3, _steps):
                self.actuator_list['z_fine'].set_parameters(step)
                new_img = np.array(self._camera.get_image(), dtype=np.float32)
                img_list.append(new_img)

            if _steps == 0:
                new_img = np.array(self._camera.get_image(), dtype=np.float32)
                img_list.append(new_img)
            
            img_arr = np.array(img_list)
            self.log.debug(f'Acquired {_steps} steps and returned {img_arr.shape} array.')

            # close the shutter (20mm = closed)
            if use_shutter:
                self.actuator_list['idler_shutter'].set_parameters(20e-3)
                self.log.debug('Closed the shutter.')

            # take an image
            if use_shutter or take_dark:
                self.dark_img = np.array(self._camera.get_image(), dtype=np.float32)

            if use_shutter:
                self.actuator_list['idler_shutter'].set_parameters(0, asynch=True)

            grp = self.save_file.create_group(f'run_{self._counter}')
            if take_dark or use_shutter:
                grp.create_dataset('dark_img', data=self.dark_img)
            grp.create_dataset('fringes', data=img_arr)
            grp.create_dataset('params', data=params)

            # plot the various images
            self.plot_queue.put(('replace', {'y': np.clip(img_arr[0], 0, 400)}, 'fringe_img'))
            self.plot_queue.put(('replace', {'y': np.clip(self.dark_img, 0, 400)}, 'dark_img'))

            self._counter += 1

            if cost_definition == 'visibility':
                return self._cost_visibility(img_arr, self.dark_img, get_scale, scale)
            elif cost_definition == 'std':
                return self._cost_std(img_arr)
            elif cost_definition == 'zernike':
                return self._cost_std(img_arr)

        # if we skip data, return None
        return None

    def _cost_visibility(self, img_in, dark_img_in, get_scale=False, scale=(1, 1)):
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

    def _cost_std(self, img_in):
        img = np.copy(img_in)
        dark_img = np.copy(self.dark_img)
        radius = 55
        center = (145, 125)

        # define the bounds
        edge_px = 5
        ymin = int(center[0] - radius - edge_px)
        ymax = int(center[0] + radius + edge_px)
        xmin = int(center[1] - radius - edge_px)
        xmax = int(center[1] + radius + edge_px)

        # crop the images
        cropped_imgs = img[:, ymin:ymax, xmin:xmax]
        cropped_dark_img = dark_img[ymin:ymax, xmin:xmax]

        cost = - np.std(cropped_imgs[0] - cropped_dark_img)
        return cost

    def _cost_zernike(self, img_in):
        img = np.copy(img_in)
        dark_img = np.copy(self.dark_img)
        radius = 55
        center = (145, 125)

        # define the bounds
        edge_px = 5
        ymin = int(center[0] - radius - edge_px)
        ymax = int(center[0] + radius + edge_px)
        xmin = int(center[1] - radius - edge_px)
        xmax = int(center[1] + radius + edge_px)

        # crop the images
        cropped_imgs = img[:, ymin:ymax, xmin:xmax]
        cropped_dark_img = dark_img[ymin:ymax, xmin:xmax]

        zdecomp = compute_zernike_decomposition(cropped_imgs[0], self.zernike_object)
        cost = cost_evaluation(zdecomp)
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