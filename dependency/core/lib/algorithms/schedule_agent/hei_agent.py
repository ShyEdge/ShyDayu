import abc
import os.path
import threading
import time
import numpy as np

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context
from core.lib.estimation import AccEstimator, OverheadEstimator
from core.lib.common import VideoOps

from .base_agent import BaseAgent

__all__ = ('HEIAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hei')
class HEIAgent(BaseAgent, abc.ABC):

    def __init__(self, system,
                 agent_id: int,
                 window_size: int = 10,
                 mode: str = 'inference',
                 model_dir: str = 'model',
                 load_model: bool = False,
                 load_model_episode: int = 0,
                 acc_gt_dir: str = '',
                 relaxed_coefficient: float = 1.6,
                 punishment_coefficient: float = 20,
                 punishment_bound: float = -2,
                 reward_bound: float = 0.5,
                 reward_coefficient: float = 0.3, ):
        from .hei import SoftActorCritic, RandomBuffer, Adapter, NegativeFeedback, StateBuffer

        self.agent_id = agent_id
        self.system = system

        drl_params = system.drl_params.copy()
        hyper_params = system.hyper_params.copy()
        drl_params['state_dims'] = [drl_params['state_dims'], window_size]

        self.window_size = window_size
        self.state_buffer = StateBuffer(self.window_size)
        self.mode = mode

        self.relaxed_coefficient = relaxed_coefficient
        self.punishment_coefficient = punishment_coefficient
        self.punishment_bound = punishment_bound
        self.reward_bound = reward_bound
        self.reward_coefficient = reward_coefficient

        self.drl_agent = SoftActorCritic(**drl_params)
        self.replay_buffer = RandomBuffer(**drl_params)
        self.adapter = Adapter

        self.nf_agent = NegativeFeedback(system, agent_id)

        self.drl_schedule_interval = hyper_params['drl_schedule_interval']
        self.nf_schedule_interval = hyper_params['nf_schedule_interval']

        self.state_dim = drl_params['state_dims']
        self.action_dim = drl_params['action_dim']

        self.acc_gt_dir = acc_gt_dir
        self.acc_estimator = None

        self.model_dir = Context.get_file_path(os.path.join('scheduler/hei', model_dir, f'agent_{self.agent_id}'))
        FileOps.create_directory(self.model_dir)
        if load_model:
            self.drl_agent.load(self.model_dir, load_model_episode)

        self.total_steps = hyper_params['drl_total_steps']
        self.save_interval = hyper_params['drl_save_interval']
        self.update_interval = hyper_params['drl_update_interval']
        self.update_after = hyper_params['drl_update_after']

        self.intermediate_decision = [0 for _ in range(self.action_dim)]

        self.latest_policy = None
        self.latest_task_delay = None
        self.schedule_plan = None

        self.reward_file = Context.get_file_path(os.path.join('scheduler/hei', 'reward.txt'))
        FileOps.remove_file(self.reward_file)

        self.macro_overhead_estimator = OverheadEstimator('HEI-Macro', 'scheduler/hei')
        self.micro_overhead_estimator = OverheadEstimator('HEI-Micro', 'scheduler/hei')

    def get_drl_state_buffer(self):
        while True:
            state, evaluation_info = self.state_buffer.get_state_buffer()
            if state is not None and evaluation_info is not None:
                return state, evaluation_info
            LOGGER.info(f'[Wait for State] (agent {self.agent_id}) State empty, '
                        f'wait for resource state or scenario state ..')
            time.sleep(1)

    def map_drl_action_to_decision(self, action):
        """
        map [-1, 1] to {-1, 0, 1}
        """

        self.intermediate_decision = [int(np.sign(a)) if abs(a) > 0.3 else 0 for a in action]

        LOGGER.info(f'[DRL Decision] (agent {self.agent_id}) Action: {action}   Decision:{self.intermediate_decision}')

    def reset_drl_env(self):
        self.intermediate_decision = [0 for _ in range(self.action_dim)]

        state, _ = self.get_drl_state_buffer()

        return state

    def step_drl_env(self, action):

        self.map_drl_action_to_decision(action)

        time.sleep(self.drl_schedule_interval)

        state, evaluation_info = self.get_drl_state_buffer()
        reward = self.calculate_drl_reward(evaluation_info)
        done = False
        info = ''

        return state, reward, done, info

    def create_acc_estimator(self, service_name: str):
        gt_path_prefix = os.path.join(self.acc_gt_dir, service_name)
        gt_file_path = Context.get_file_path(os.path.join(gt_path_prefix, 'gt_file.txt'))
        self.acc_estimator = AccEstimator(gt_file_path)

    def calculate_drl_reward(self, evaluation_info):
        delay_bias_list = []
        acc_list = []

        for task in evaluation_info:
            delay = task.calculate_total_time()
            meta_data = task.get_metadata()
            raw_metadata = task.get_raw_metadata()
            content = task.get_content()
            pipeline = task.get_pipeline()

            hash_data = task.get_hash_data()

            raw_resolution = VideoOps.text2resolution(raw_metadata['resolution'])
            resolution = VideoOps.text2resolution(meta_data['resolution'])
            resolution_ratio = (resolution[0] / raw_resolution[0], resolution[1] / raw_resolution[1])

            fps_ratio = meta_data['fps'] / raw_metadata['fps']

            if not self.acc_estimator:
                self.create_acc_estimator(service_name=pipeline[0].get_service_name())
            acc = self.acc_estimator.calculate_accuracy(hash_data, content, resolution_ratio, fps_ratio)
            acc_list.append(acc)

            single_task_delay = delay / meta_data['buffer_size']
            single_task_constraint = 1 / meta_data['fps']
            delay_bias_list.append(single_task_constraint * self.relaxed_coefficient - single_task_delay)

        final_delay = np.mean(delay_bias_list)
        final_acc = np.mean(acc_list)
        LOGGER.info(f'[Reward Computing] delay:{final_delay} acc:{final_acc}')

        if final_delay < 0:
            reward = max(final_delay * self.punishment_coefficient, self.punishment_bound)
        else:
            reward = 1 / max(final_delay, self.reward_bound) * self.reward_coefficient + final_acc

        with open(self.reward_file, 'a') as f:
            f.write(f'delay:{final_delay} acc:{final_acc} reward:{reward}\n')

        return reward

    def train_drl_agent(self):
        LOGGER.info(f'[DRL Train] (agent {self.agent_id}) Start train drl agent ..')
        state = self.reset_drl_env()
        for step in range(self.total_steps):

            with self.macro_overhead_estimator:
                action = self.drl_agent.select_action(state, deterministic=False, with_logprob=False)

            next_state, reward, done, info = self.step_drl_env(action)
            done = self.adapter.done_adapter(done, step)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            LOGGER.info(f'[DRL Train Data] (agent {self.agent_id}) Step:{step}  Reward:{reward}')

            if step >= self.update_after and step % self.update_interval == 0:
                for _ in range(self.update_interval):
                    LOGGER.info(f'[DRL Train] (agent {self.agent_id}) Train drl agent with replay buffer')
                    self.drl_agent.train(self.replay_buffer)

            if step % self.save_interval == 0:
                self.drl_agent.save(self.model_dir, step)

            if done:
                state = self.reset_drl_env()

        LOGGER.info(f'[DRL Train] (agent {self.agent_id}) End train drl agent ..')

    def inference_drl_agent(self):
        LOGGER.info(f'[DRL Inference] (agent {self.agent_id}) Start inference drl agent ..')
        state = self.reset_drl_env()
        cur_step = 0
        while True:
            time.sleep(self.drl_schedule_interval)
            cur_step += 1

            with self.macro_overhead_estimator:
                action = self.drl_agent.select_action(state, deterministic=False, with_logprob=False)

            next_state, reward, done, info = self.step_drl_env(action)
            done = self.adapter.done_adapter(done, cur_step)
            state = next_state
            if done:
                state = self.reset_drl_env()
                cur_step = 0

    def run_nf_agent(self):
        LOGGER.info(f'[NF Inference] (agent {self.agent_id}) Start inference nf agent ..')

        while True:
            time.sleep(self.nf_schedule_interval)

            with self.micro_overhead_estimator:
                self.schedule_plan = self.nf_agent(self.latest_policy,
                                                   self.latest_task_delay,
                                                   self.intermediate_decision)

            LOGGER.debug(f'[NF Update] (agent {self.agent_id}) schedule: {self.schedule_plan}')

    def update_scenario(self, scenario):
        try:
            object_number = np.mean(scenario['obj_num'])
            object_size = np.mean(scenario['obj_size'])
            task_delay = scenario['delay']

            self.latest_task_delay = task_delay

            self.state_buffer.add_scenario_buffer([object_number, object_size, task_delay])
        except Exception as e:
            LOGGER.warning(f'Wrong scenario from Distributor: {str(e)}')

    def update_resource(self, device, resource):
        bandwidth = resource['bandwidth']
        if bandwidth != 0:
            self.state_buffer.add_resource_buffer([bandwidth])

    def update_policy(self, policy):
        self.set_latest_policy(policy)

        resolution_decision = self.system.resolution_list.index(policy['resolution'])
        fps_decision = self.system.fps_list.index(policy['fps'])
        buffer_size_decision = self.system.buffer_size_list.index(policy['buffer_size'])
        pipeline_decision = next((i for i, service in enumerate(policy['pipeline'])
                                  if service['execute_device'] == self.system.cloud_device),
                                 len(policy['pipeline']) - 1)
        self.state_buffer.add_decision_buffer([resolution_decision, fps_decision,
                                               buffer_size_decision, pipeline_decision])

    def update_task(self, task):
        self.state_buffer.add_task_buffer(task)

    def set_latest_policy(self, policy):
        self.latest_policy = policy

    def get_schedule_plan(self, info):
        return self.schedule_plan

    def run(self):
        threading.Thread(target=self.run_nf_agent).start()

        if self.mode == 'train':
            self.train_drl_agent()
        elif self.mode == 'inference':
            self.inference_drl_agent()
        else:
            assert None, f'(agent {self.agent_id}) Invalid execution mode: {self.mode}, ' \
                         f'only support ["train", "inference"]'
