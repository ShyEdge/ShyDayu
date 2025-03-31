import threading
import torch # type: ignore
import torch.nn as nn
import torch.nn.functional as F  # type: ignore
import numpy as np   # type: ignore
import random
from . import rl_utils
from collections import deque

'''
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
'''

class StateBuffer:
    def __init__(self, maxlen=8):
        self.maxlen = maxlen 
        
        self.cpu_local = deque(maxlen=maxlen)
        self.cpu_other = deque(maxlen=maxlen)
        self.bandwidth_edge_local = deque(maxlen=maxlen)
        self.bandwidth_edge_other = deque(maxlen=maxlen)
        self.last_decision = deque(maxlen=maxlen)
        self.last_delay = deque(maxlen=maxlen)  #这三个都从scenario中获取
        self.last_task_obj_num = deque(maxlen=maxlen)
        self.last_task_obj_size = deque(maxlen=maxlen)

        for _ in range(maxlen):
            self.cpu_local.append(0)
            self.cpu_other.append(0)
            self.bandwidth_edge_local.append(0)
            self.bandwidth_edge_other.append(0)
            self.last_decision.append(0)
            self.last_delay.append(0)
            self.last_task_obj_num.append(0)  
            self.last_task_obj_size.append(0)  

    def update(self, cpu_local, cpu_other, bandwidth_local, bandwidth_other, decision, delay, obj_num, obj_size):
        """更新状态"""
        self.cpu_local.append(cpu_local)
        self.cpu_other.append(cpu_other)
        self.bandwidth_edge_local.append(bandwidth_local)
        self.bandwidth_edge_other.append(bandwidth_other)
        self.last_decision.append(decision)
        self.last_delay.append(delay)
        self.last_task_obj_num.append(obj_num)  
        self.last_task_obj_size.append(obj_size)  

    def get_state_vector(self):
        """将 deque 数据转换为 2D NumPy 数组"""
        return np.array([
            list(self.cpu_local),
            list(self.cpu_other),
            list(self.bandwidth_edge_local),
            list(self.bandwidth_edge_other),
            list(self.last_decision),
            list(self.last_delay),
            list(self.last_task_obj_num),
            list(self.last_task_obj_size)
        ])

    def get_state_shape(self):
        """返回状态向量的形状"""
        return self.get_state_vector().shape



class PolicyNet(nn.Module):
    def __init__(self, state_channels, history_length, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # 1D卷积层：保持时间步数不变
        self.conv1 = nn.Conv1d(in_channels=state_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 经过两个卷积层后，输出形状为 [batch_size, 32, history_length]
        # 将卷积输出展平后送入全连接层
        self.fc = nn.Linear(32 * history_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # x 的输入形状为 [batch_size, channels, history_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        return F.softmax(self.out(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_channels, history_length, hidden_dim):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=state_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * history_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)


class ActorCritic:
    def __init__(self, state_channels, history_length, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_channels, history_length, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_channels, history_length, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)   # 动作概率分布

        print("------------------------probs是------------------------------")
        print(probs)
        print("-------------------------------------------------------------")

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
        
        print("*****************************************************update content*****************************************************")
        print(f"states是{states}")
        print(f"actions是{actions}")
        print(f"rewards是{rewards}")
        #print(f"dones是{dones}")
        print("************************************************************************************************************************")

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))  #在维度1（列方向）选取actions指定的索引对应的概率，即实际执行的每个动作的概率
        actor_loss = torch.mean(-log_probs * td_delta.detach())  #detach() 用于 阻断梯度传播，防止 TD 误差影响 Critic 网络 的参数更新，保证Critic仅由critic_loss 进行更新
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))  #td_target 本身是由 Critic 计算出来的，所以它 包含 Critic 网络的计算图，如果不 detach()，它的梯度可能会影响 Critic 网络的学习，我们希望让td_target.detach()只是个常量
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward() 
        critic_loss.backward() 
        self.actor_optimizer.step() 
        self.critic_optimizer.step() 




class CloudEdgeEnv():
    def __init__(self, device_info=None, cloud_device=None):
        self.device_info = device_info
        self.device_list = list(self.device_info.values()) + [cloud_device]  #按yaml顺序
        self.local_edge = self.device_list[0]
        self.other_edges = self.device_list[1:-1]

        self.device_info['cloud'] = cloud_device

        self.resource_table = None
        self.train_parameters = None
        self.selected_device = [cloud_device, cloud_device]

        self.condition = threading.Condition()

        self.scenario = None

        self.task_count = 0
        self.max_count = 30

        self.reward_list = []
        self.reward_list_avg = []
        self.delay_list = []
        self.delay_list_avg = []

        self.state_buffer = StateBuffer()


    def reset(self):
        return self.state_buffer.get_state_vector()


    def step(self, action):  #执行一个动作并返回环境的下一个状态、奖励、是否完成以及附加信息    
        '''
        if action == self.action_space_n - 1:
            self.selected_device = [self.device_info['cloud'], self.device_info['cloud']]
        else:
            # action 对 x 取模后，选择对应的设备
            x = len(self.device_info)
            idx1 = action // x  
            idx2 = action % x  
            self.selected_device = [self.device_list[idx1], self.device_list[idx2]]
        '''
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

        #print("*****************************************************drl step wait for condition*****************************************************")
        with self.condition: 
            self.condition.notify_all() 
            self.condition.wait()  
        #print("*****************************************************drl step wait for condition end*************************************************")
        
        
        new_state = self.get_new_state(action) 

        #避免none导致的报错
        if self.scenario is None:
            self.scenario = {
                "delay": 1,
                "obj_num": [0],
                "obj_size": [0]
            }

        reward = self.compute_reward(self.scenario['delay'])

        self.display_rewards(self.scenario['delay'], reward)

        done = self.check_done()    #执行指定数量的task后作为结束标志

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

    def get_new_state(self, action):
        cpu_local, cpu_other = self.extract_cpu_state()
        bandwidth_local, bandwidth_other = self.extract_bandwidth_state()

        #避免none导致的报错
        if self.scenario is None:
            self.scenario = {
                "delay": 1,
                "obj_num": [0],
                "obj_size": [0]
            }

        delay = self.scenario.get('delay', 1) or 1
        obj_num = self.scenario.get('obj_num', [0]) or [0]
        obj_size = self.scenario.get('obj_size', [0]) or [0]

        obj_num_avg = np.mean(obj_num) if len(obj_num) > 0 else 0
        obj_size_avg = np.mean(obj_size) if len(obj_size) > 0 else 0
        
        self.state_buffer.update(cpu_local, cpu_other, bandwidth_local, bandwidth_other, action, delay, obj_num_avg, obj_size_avg)

        return self.state_buffer.get_state_vector()

    def extract_cpu_state(self):

        local_edge_cpu = None  
        other_edge_cpu = None  

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

        return bandwidth_value, bandwidth_value
    
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
        
        local_edge_cpu, other_edge_cpu = self.extract_cpu_state()
        load_balance = abs(local_edge_cpu - other_edge_cpu)

        alpha = self.train_parameters['reward_alpha']
        beta = self.train_parameters['reward_beta']
        c = self.train_parameters['reward_c']

        reward = -alpha * delay - beta * load_balance + c

        return reward


def train_actorcritic_on_policy(env):
    
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98  # 奖励折扣
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_channels = 8
    history_length = 8
    action_dim = 6
    
    agent = ActorCritic(state_channels, history_length, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device)

    rl_utils.train_on_policy_agent_CloudEdge(env, agent, num_episodes)