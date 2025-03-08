import abc
from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent
from .actor_critic.network import CloudEdgeEnv

__all__ = ('ActorCriticAgent',)




@ClassFactory.register(ClassType.SCH_AGENT, alias='actorcritic')
class ActorCriticAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, actorcritic_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.actorcritic_policy = actorcritic_policy
        #self.services_allocate = Context.get_algorithm('SCH_SERVICES_ALLOCATE')
        self.env = CloudEdgeEnv(actorcritic_policy['device_info'], system.cloud_device)
        

    def get_schedule_plan(self, info):
   
        if self.actorcritic_policy is None:
            return self.actorcritic_policy  #直接返回None

        policy = self.actorcritic_policy.copy()

        local_device = info['device']
        pipeline = info['pipeline']
        resource_table = info['resource_table']

        cloud_device = self.cloud_device

        #设备信息
        device_info = policy['device_info']
        device_info['cloud'] = cloud_device
        device_info['local'] = local_device

        #训练需要的参数
        train_para = policy['train_parameters']
        
        #更新资源表信息
        self.env.update_resource_table(resource_table)




        #pipeline=self.services_allocate(decision_para, device_info, pipeline, resource_table)

        policy.update({'pipeline': pipeline})

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
