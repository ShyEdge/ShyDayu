import abc
import random

from core.lib.common import ClassFactory, ClassType
from .base_devices_allocate import BaseDevicesAllocate


@ClassFactory.register(ClassType.SCH_DEVICES, alias='simple')
class SimpleDevicesAllocate(BaseDevicesAllocate ,abc.ABC):

    def __call__(self, allo_num_of_devices,num_of_devices):
        return self.generate_random_indexes(allo_num_of_devices,num_of_devices)
    
    #注意最后要考虑云只能不存在或只在最后出现！
    def generate_random_indexes(self,allo_num_of_devices,num_of_devices):
        random_indexes=random.sample(range(num_of_devices), allo_num_of_devices)
        return random_indexes          