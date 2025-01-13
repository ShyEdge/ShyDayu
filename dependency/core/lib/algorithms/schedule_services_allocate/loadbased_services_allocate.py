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

            '''
            print(f"i的内容是:{i}")
            print(f"device_ids内容是{device_ids}")
            print(f"device_info的内容是{device_info}")
            print(f"device_ids[i]的内容是:{device_ids[i]}")
            print(f"device_info[device_ids[i]]的内容是{device_info[device_ids[i]]}")
            '''

            execute_device = device_info[device_ids[i]]   

            for p in pipeline[start:end]:
                distributed_pipeline.append({**p, 'execute_device': execute_device})

        return distributed_pipeline   