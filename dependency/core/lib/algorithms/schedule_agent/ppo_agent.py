import abc
from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent
from .ppo.network import CloudEdgeEnv, train_ppo_on_policy

__all__ = ('PPOAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='ppo')
class PPOAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, ppo_policy: dict = None):
        self.agent_id = agent_id
        self.ppo_policy = ppo_policy
        self.last_task_scenario = None
        self.env = CloudEdgeEnv(ppo_policy['device_info'], system.cloud_device)
        self.env.set_train_parameters(ppo_policy['train_parameters'])

    def get_schedule_plan(self, info):
        if self.ppo_policy is None:
            return self.ppo_policy  #直接返回None
        policy = self.ppo_policy.copy()

        local_device = info['device']
        self.env.set_local_edge(local_device)  

        pipeline = info['pipeline']
        resource_table = info['resource_table']
                
        self.env.update_resource_table(resource_table)  
        self.env.update_scenario(self.last_task_scenario)  
      
        with self.env.condition:
            self.env.condition.notify_all()  
            self.env.condition.wait()  
        
        execute_device = self.env.get_selected_device()
        
        # 修改Pipeline的内容，注意后者必须用[:1]这种，而不能用[0]，会报错
        pipeline = [{**p, 'execute_device': execute_device[0]} for p in pipeline[:1]] + \
           [{**p, 'execute_device': execute_device[1]} for p in pipeline[1:]]


        policy.update({'pipeline': pipeline})

        return policy


    def run(self):
        train_ppo_on_policy(self.env)

    def update_scenario(self, scenario):
        self.last_task_scenario = scenario

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass
