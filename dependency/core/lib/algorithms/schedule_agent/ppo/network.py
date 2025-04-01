import torch
import torch.nn.functional as F
import numpy as np
import threading
import random
from collections import deque
from . import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)

        print("--------------------------------------------------------------")
        print(f"probs is {probs}")
        print("--------------------------------------------------------------")

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        

        print("--------------------------------------------------------------")
        print(f"states is {states}")
        print(f"actions is {actions}")
        print(f"rewards is {rewards}")
        print("--------------------------------------------------------------")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class StateBuffer:
    def __init__(self, maxlen=1):
        self.maxlen = maxlen 
        self.bandwidth = deque(maxlen=maxlen)

        for _ in range(maxlen):
            self.bandwidth.append(0)

    def update(self, bandwidth):
        self.bandwidth.append(bandwidth)


    def get_state_vector(self):
        return np.array([
            list(self.bandwidth),
        ]).flatten()


class CloudEdgeEnv():
    def __init__(self, device_info=None, cloud_device=None):
        self.device_info = device_info
        self.device_list = list(self.device_info.values()) + [cloud_device]  #按yaml顺序
        self.local_edge = self.device_list[0]
        self.other_edges = self.device_list[1:-1]
        self.device_info['cloud'] = cloud_device

        self.resource_table = None
        self.train_parameters = None
        self.scenario = None

        self.selected_device = [cloud_device, cloud_device]

        self.condition = threading.Condition()

        self.task_count = 0
        self.max_count = 50

        self.reward_list = []
        self.reward_list_avg = []
        self.delay_list = []
        self.delay_list_avg = []

        self.state_buffer = StateBuffer()

    def reset(self):
        return self.state_buffer.get_state_vector()

    def step(self, action):   

        selected_edge = random.choice(self.other_edges)
        if action == 0:
            self.selected_device = [self.local_edge, self.local_edge]
        elif action == 1:
            self.selected_device = [self.local_edge, selected_edge]
        elif action == 2:
            self.selected_device = [self.local_edge, self.device_info['cloud']]
        elif action == 3:
            self.selected_device = [selected_edge, selected_edge]
        elif action == 4:
            self.selected_device = [selected_edge, self.device_info['cloud']]
        elif action == 5:
            self.selected_device = [self.device_info['cloud'], self.device_info['cloud']]
        else:
            raise ValueError("Invalid action")

        with self.condition: 
            self.condition.notify_all() 
            self.condition.wait()    
        
        new_state = self.get_new_state() 

        if self.scenario is None:
            self.scenario = {
                "delay": 1,
                "obj_num": [0],
                "obj_size": [0]
            }

        reward = self.compute_reward(self.scenario['delay'])

        self.display_rewards(self.scenario['delay'], reward)

        done = self.check_done()   

        return new_state, reward, done, {}  

    def update_resource_table(self, resource_table):   
        self.resource_table = resource_table

    def update_scenario(self, scenario):
        self.scenario = scenario

    def set_local_edge(self, local_edge):
        self.local_edge = local_edge

    def get_selected_device(self):
        return self.selected_device
    
    def set_train_parameters(self, train_parameters):
        self.train_parameters = train_parameters

    def get_new_state(self):
        bandwidth = self.extract_bandwidth_state()
        self.state_buffer.update(bandwidth)
        return self.state_buffer.get_state_vector()

    def extract_cpu_state(self):
        local_edge_cpu = 50
        other_edge_cpu = 50

        for key, val in self.resource_table.items():
            if key.startswith('edge'):
                if 'cpu' in val:
                    if key == self.local_edge:
                        local_edge_cpu = val['cpu']
                    else:
                        other_edge_cpu = val['cpu']

        return local_edge_cpu, other_edge_cpu
 
    def extract_bandwidth_state(self):
        bandwidth_value = None

        for key, val in self.resource_table.items():
            if 'bandwidth' in val:
                bandwidth_value = val['bandwidth']
                if key.startswith('edge'):  # 如果是 edge，立即返回
                    print(key, bandwidth_value)
                    break

        return bandwidth_value
    
    def check_done(self):
        self.task_count += 1
        done = self.task_count >= self.max_count
        if done:
            self.task_count = 0
        return done
    
    def display_rewards(self, delay, reward):
        self.delay_list.append(delay)
        self.reward_list.append(reward)
        list_length = len(self.reward_list)
    
        if list_length % self.max_count == 0:
            avg_reward = sum(self.reward_list[-self.max_count:]) / self.max_count  
            self.reward_list_avg.append(avg_reward)
            print(f"reward_list_avg的内容是{self.reward_list_avg}")

            avg_delay = sum(self.delay_list[-self.max_count:]) / self.max_count  
            self.delay_list_avg.append(avg_delay)
            print(f"delay_list_avg的内容是{self.delay_list_avg}")

    def compute_reward(self, delay):
        c = self.train_parameters['reward_c']
        reward = c - delay
        return max(reward, -3)  #clip   



def train_ppo_on_policy(env):
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    state_dim = 1
    action_dim = 6

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

    rl_utils.train_on_policy_agent_CloudEdge(env, agent, num_episodes)


