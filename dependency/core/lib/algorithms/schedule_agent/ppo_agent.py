import abc, time
from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent
from .ppo.network import CloudEdgeEnv, train_ppo_on_policy

__all__ = ('PPOAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='ppo')
class PPOAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, ppo_policy: dict = None):
        self.agent_id = agent_id
        self.ppo_policy = ppo_policy
        self.env = CloudEdgeEnv(ppo_policy['device_info'], system.cloud_device)
        self.env.set_train_parameters(ppo_policy['train_parameters'])

    def get_schedule_plan(self, info):
        if self.ppo_policy is None:
            return self.ppo_policy  #直接返回None
        policy = self.ppo_policy.copy()

        local_device = info['device']
        self.env.set_local_edge(local_device)  

        pipeline = info['pipeline']
                        
        execute_device, selected_action = self.env.get_selected_device()
        self.env.update_decision(selected_action)  #TODO 目前选择的更新时间并不合理
        
        # 修改Pipeline的内容，注意后者必须用[:1]这种，而不能用[0]，会报错
        pipeline = [{**p, 'execute_device': execute_device[0]} for p in pipeline[:1]] + \
           [{**p, 'execute_device': execute_device[1]} for p in pipeline[1:]]

        policy.update({'pipeline': pipeline})

        return policy


    def run(self):
        time.sleep(10)
        train_ppo_on_policy(self.env)

    def update_scenario(self, scenario):
        if 'delay' in scenario:
            self.env.update_delay(scenario['delay'])
        if 'obj_num' in scenario:
            self.env.update_task_obj_num(scenario['obj_num'])
        if 'obj_size' in scenario:
            self.env.update_task_obj_size(scenario['obj_size'])

    def update_resource(self, device, resource, resource_table):
        self.env.update_resource_state(resource_table)

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass
