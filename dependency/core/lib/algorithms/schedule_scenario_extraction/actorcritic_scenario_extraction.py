import abc

from core.lib.common import ClassFactory, ClassType
from .base_scenario_extraction import BaseScenarioExtraction

__all__ = ('ActorCriticScenarioExtraction',)


@ClassFactory.register(ClassType.SCH_SCENARIO, alias='actorcritic')
class ActorCriticScenarioExtraction(BaseScenarioExtraction, abc.ABC):
    def __call__(self, task):
        scenario = task.get_scenario_data()
        delay = task.calculate_total_time()
        scenario['delay'] = delay
        return scenario
