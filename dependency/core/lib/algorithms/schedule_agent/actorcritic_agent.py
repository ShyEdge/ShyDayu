import abc

from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent
from .actor_critic.network import CloudEdgeEnv, train_actorcritic_on_policy

__all__ = ('ActorCriticAgent',)




@ClassFactory.register(ClassType.SCH_AGENT, alias='actorcritic')
class ActorCriticAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, actorcritic_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.actorcritic_policy = actorcritic_policy
        self.last_task_scenario = None
        self.env = CloudEdgeEnv(actorcritic_policy['device_info'], system.cloud_device)
        self.env.set_train_parameters(actorcritic_policy['train_parameters'])

    def get_schedule_plan(self, info):
    
        if self.actorcritic_policy is None:
            return self.actorcritic_policy  #直接返回None

        policy = self.actorcritic_policy.copy()

        local_device = info['device']
        self.env.set_local_edge(local_device)  

        pipeline = info['pipeline']
        resource_table = info['resource_table']

        print("--------------------resource_tables是-------------------------------------")
        print(resource_table)
        print("--------------------------------------------------------------------------")

        cloud_device = self.cloud_device
        #设备信息
        device_info = policy['device_info']
        device_info['cloud'] = cloud_device
        

        
        self.env.update_resource_table(resource_table)  #next_state
        self.env.update_scenario(self.last_task_scenario)  #reward

        #print("--------------------------------------------------------sch_agent wait for condition--------------------------------------------------------")
        with self.env.condition:
            self.env.condition.notify_all()  
            self.env.condition.wait()  
        #print("--------------------------------------------------------sch_agent wait for condition end----------------------------------------------------")

        execute_device = self.env.get_selected_device()
        
        # 修改Pipeline的内容，注意后者必须用[:1]这种，而不能用[0]，会报错
        pipeline = [{**p, 'execute_device': execute_device[0]} for p in pipeline[:1]] + \
           [{**p, 'execute_device': execute_device[1]} for p in pipeline[1:]]


        policy.update({'pipeline': pipeline})

        return policy


    def run(self):
        train_actorcritic_on_policy(self.env)

    def update_scenario(self, scenario):
        self.last_task_scenario = scenario

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass
