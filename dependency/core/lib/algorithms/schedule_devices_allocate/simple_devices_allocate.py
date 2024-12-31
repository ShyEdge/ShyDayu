import abc
import random

from core.lib.common import ClassFactory, ClassType
from .base_devices_allocate import BaseDevicesAllocate


@ClassFactory.register(ClassType.SCH_DEVICES_ALLOCATE, alias='simple')
class SimpleDevicesAllocate(BaseDevicesAllocate ,abc.ABC):

    def __init__(self):
        pass

    def __call__(self, device_info):
        return list(device_info.keys())
    

    #def generate_random_indexes(self,allo_num_of_devices,num_of_devices):
    #    random_indexes=random.sample(range(num_of_devices), allo_num_of_devices)
    #    return random_indexes          