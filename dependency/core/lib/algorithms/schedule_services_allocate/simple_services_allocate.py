import abc
import random
#from core.lib.algorithms.schedule_devices_allocate import SimpleDevicesAllocate

from core.lib.common import ClassFactory, ClassType, Context
from .base_services_allocate import BaseServicesAllocate



@ClassFactory.register(ClassType.SCH_SERVICES_ALLOCATE, alias='simple')
class SimpleServicesAllocate(BaseServicesAllocate, abc.ABC):

    def __init__(self):
        self.devices_allocate = Context.get_algorithm("SCH_DEVICES_ALLOCATE")

    def __call__(self, pipe_segs, device_info, pipeline):

        device_ids=self.devices_allocate(device_info)
          
        segs = [0] + pipe_segs + [len(pipeline)] 
        segs = list(dict.fromkeys(segs))

        distributed_pipeline = []

        for i in range(len(segs) - 1):
            start = segs[i]  
            end = segs[i + 1]  

            execute_device = device_info[device_ids[i]]   

            for p in pipeline[start:end]:
                distributed_pipeline.append({**p, 'execute_device': execute_device})

        return distributed_pipeline    
        
        
  
       
    









