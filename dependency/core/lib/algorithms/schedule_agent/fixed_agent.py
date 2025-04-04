import abc
from core.lib.common import ClassFactory, ClassType

from .base_agent import BaseAgent

__all__ = ('FixedAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='fixed')
class FixedAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, fixed_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.fixed_policy = fixed_policy
        self.last_delay = 0
        self.sum_delay = 0
        self.delay_list = []
        self.cnt = 50

    def get_schedule_plan(self, info):
        if self.fixed_policy is None:
            return self.fixed_policy

        policy = self.fixed_policy.copy()
        edge_device = info['device']
        cloud_device = self.cloud_device
        pipe_seg = policy['pipeline']
        pipeline = info['pipeline']
        
        pipeline = [{**p, 'execute_device': edge_device} for p in pipeline[:pipe_seg]] + \
                   [{**p, 'execute_device': cloud_device} for p in pipeline[pipe_seg:]]
            
        self.sum_delay += self.last_delay
        self.cnt -= 1

        if self.cnt == 0:    
            self.delay_list.append(self.sum_delay/50)
            self.cnt = 50
            self.sum_delay = 0
            print(self.delay_list)

        policy.update({'pipeline': pipeline})
        return policy

    def run(self):
        pass

    def update_scenario(self, scenario):
        if scenario is None:
            self.last_delay = 1  
            return 
        
        if 'delay' not in scenario:
            self.last_delay = 1
            return 
            
        self.last_delay = scenario['delay']


    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass
