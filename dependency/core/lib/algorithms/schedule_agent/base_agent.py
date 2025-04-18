import abc


class BaseAgent(metaclass=abc.ABCMeta):
    def __call__(self):
        raise NotImplementedError

    def update_scenario(self, scenario):
        raise NotImplementedError

    def update_resource(self, device, resource):
        raise NotImplementedError

    def update_policy(self, policy):
        raise NotImplementedError

    def update_task(self, task):
        raise NotImplementedError

    def get_schedule_plan(self, info):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


