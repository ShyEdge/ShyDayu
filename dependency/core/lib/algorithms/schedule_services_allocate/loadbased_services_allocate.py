import abc

from core.lib.common import ClassFactory, ClassType, Context
from .base_services_allocate import BaseServicesAllocate



@ClassFactory.register(ClassType.SCH_SERVICES_ALLOCATE, alias='loadbased')
class LoadBasedServicesAllocate(BaseServicesAllocate, abc.ABC):

    def __init__(self):
        self.devices_allocate = Context.get_algorithm("SCH_DEVICES_ALLOCATE")

    def __call__(self, decision_para, device_info, pipeline, resource_table):
   
        #segs = [0]  + [len(pipeline)] 
        #segs = list(dict.fromkeys(segs))

        device_ids, segs = self.devices_allocate(decision_para, device_info, pipeline, resource_table)

        distributed_pipeline = []

        for i in range(len(segs) - 1):
            start = segs[i]  
            end = segs[i + 1]  

            execute_device = device_info[device_ids[i]]   

            for p in pipeline[start:end]:
                distributed_pipeline.append({**p, 'execute_device': execute_device})

        return distributed_pipeline   