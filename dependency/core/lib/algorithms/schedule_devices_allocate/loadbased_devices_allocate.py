import abc
import random

from core.lib.common import ClassFactory, ClassType
from .base_devices_allocate import BaseDevicesAllocate

@ClassFactory.register(ClassType.SCH_DEVICES_ALLOCATE, alias='loadbased')
class LoadBasedDevicesAllocate(BaseDevicesAllocate ,abc.ABC):
    def __init__(self):
        pass

    def __call__(self, decision_para, device_info, pipeline, resource_table):
     
        #源设备
        local_device = device_info['device_local']
        
        #其它可用的边缘设备
        other_device = {
            key: value
            for key, value in device_info.items()
            if value != local_device and key != 'device_cloud'
        }

        selected_device = random.choice(list(other_device.keys()))
        segs = [0]
        index = 0
        device_id=['device_local']
        

        if (resource_table[local_device]['cpu'] < decision_para['threshold_local']):  
            return device_id, segs + [len(pipeline)]

        #从可用边缘设备中任选一个设备，边-边
        elif (resource_table[other_device[selected_device]]['cpu'] < decision_para['threshold_other']):  #如果只有一个边缘设备则不适用，因为没有selected_device
            index = index + 1
            segs.append(index)
            device_id.append(selected_device)
            return device_id, segs + [len(pipeline)]
        
        #边-云
        else:
            index = index + 1
            segs.append(index)
            device_id.append('device_cloud')
            return device_id, segs + [len(pipeline)]
            