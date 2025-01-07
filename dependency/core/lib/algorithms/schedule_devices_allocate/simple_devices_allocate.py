import abc
import random

from core.lib.common import ClassFactory, ClassType
from .base_devices_allocate import BaseDevicesAllocate


@ClassFactory.register(ClassType.SCH_DEVICES_ALLOCATE, alias='simple')
class SimpleDevicesAllocate(BaseDevicesAllocate ,abc.ABC):

    def __init__(self):
        pass

    def __call__(self, device_info, segs):

        device_ids=list(device_info.keys())
        
        device_ids[len(segs)-2]= random.choice([device_ids[len(segs)-2], device_ids[-1]])

        device_ids=device_ids[:len(segs)-1]

        return device_ids
    
    #def generate_random_indexes(self,allo_num_of_devices,num_of_devices):
    #    random_indexes=random.sample(range(num_of_devices), allo_num_of_devices)
    #    return random_indexes          