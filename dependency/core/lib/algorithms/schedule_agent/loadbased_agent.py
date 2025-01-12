import abc
from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent

__all__ = ('LoadBasedAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='loadbased')
class LoadBasedAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, loadbased_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.loadbased_policy = loadbased_policy
        self.services_allocate = Context.get_algorithm('SCH_SERVICES_ALLOCATE')

    def get_schedule_plan(self, info):
   
        if self.loadbased_policy is None:
            return self.loadbased_policy

        policy = self.loadbased_policy.copy()
        #edge_device = info['device']
        local_device = info['device']

        cloud_device = self.cloud_device

        #设备信息
        device_info= policy['device_info']
        device_info['device_cloud'] = cloud_device
        device_info['device_local'] = local_device

        #做决策需要的参数
        decision_para = policy['Decision_Parameters']

        pipeline = info['pipeline']
        resource_table = info['resource_table']

        pipeline=self.services_allocate(decision_para, device_info, pipeline, resource_table)

        policy.update({'pipeline': pipeline})

        
        print("-----------------------------------------------------------------------------------------------")
        print("LoadBasedAgent中的policy是: ",policy)
        print("-----------------------------------------------------------------------------------------------")

        return policy

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass
