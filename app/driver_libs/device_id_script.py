import os
import json
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(name)s:%(module)s:%(message)s')

_LOG = logging.getLogger('main')

def filter(prefix, dev_list):
    filtered_devs = []
    for dev in dev_list:
        if prefix in dev:
            filtered_devs.append(dev)
    return filtered_devs

def dev_diff(dev_list1, dev_list2):
    difference = []
    for dev in dev_list2:
        if dev not in dev_list1:
            difference.append(dev)

    return difference

def get_devs(prefix='ttyUSB'):
    files = os.listdir('/dev/')
    return filter(prefix, files)

actuator_list = {'longitudinal_4485': None,
                 'lateral_6514': None,
                 'sig_arm_horz_8632': None,
                 'sig_arm_vert_2363': None,
                 'z_coarse_zaber': None,
                 'z_fine_jena': None
                 }

for dev_string in actuator_list.keys():
    while True:
        input(f'> Turn off {dev_string}')
        list_1 = get_devs()
        _LOG.debug(list_1)
        input(f'> Turn on {dev_string}')
        list_2 = get_devs()
        _LOG.debug(list_2)
        differences = dev_diff(list_1, list_2)

        if len(differences) == 0:
            _LOG.error(f'Could not find device {dev_string}')
        else:
            _LOG.info(f'Found: {differences[0]}')
            print('-'*50)
            actuator_list[dev_string] = differences[0]
            break

_LOG.info('Dumping config')
with open('./device_locs.txt', 'w+') as f:
    json.dump(actuator_list, f)

_LOG.info('All done!')