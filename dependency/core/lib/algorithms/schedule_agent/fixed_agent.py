import abc
from core.lib.common import ClassFactory, ClassType, Context

from .base_agent import BaseAgent

__all__ = ('FixedAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='fixed')
class FixedAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, fixed_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.fixed_policy = fixed_policy
        self.services_allocate = Context.get_algorithm('SCH_SERVICES_ALLOCATE')

    def get_schedule_plan(self, info):
        if self.fixed_policy is None:
            return self.fixed_policy

        policy = self.fixed_policy.copy()
        edge_device = info['device']
        cloud_device = self.cloud_device

        device_info={
            "device_1": edge_device,
            "device_2": cloud_device,
        }

        pipe_segs= policy['pipeline']
        pipeline = info['pipeline']
        pipeline=self.services_allocate(pipe_segs, device_info, pipeline)

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
