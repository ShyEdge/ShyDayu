import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, time
from collections import deque, OrderedDict
from . import rl_utils


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


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, history_length, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, history_length, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, history_length, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

        #self.actor_loss_list = []
        #self.critic_loss_list = []
        #self.entropy_list = []


    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)

        print("--------------------------------------------------------------")
        print(f"probs is {probs}")
        print("--------------------------------------------------------------")

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        
        return action

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
        
        print(f"states is {states}")
        print(f"actions is {actions}")
        print(f"rewards is {rewards}")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        # beta = 0.1  # 熵正则项系数

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数

            # 计算策略熵并加入损失
            # entropy = torch.distributions.Categorical(self.actor(states)).entropy().mean()
            # actor_loss -= beta * entropy

            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            
            '''
            if _ == 0:
                #self.entropy_list.append(entropy.item())
                self.actor_loss_list.append(actor_loss.item())
                self.critic_loss_list.append(critic_loss.item())    

                #print(f"entropy_list is {self.entropy_list}")
                print(f"actor_loss_list is {self.actor_loss_list}")
                print(f"critic_loss_list is {self.critic_loss_list}")
            '''

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        

class StateBuffer:
    def __init__(self, maxlen=3):
        self.maxlen = maxlen 
        
        self.cpu_local = deque(maxlen=maxlen)
        self.cpu_other = deque(maxlen=maxlen)
        self.bandwidth_edge_local = deque(maxlen=maxlen)
        self.bandwidth_edge_other = deque(maxlen=maxlen)
        self.last_decision = deque(maxlen=maxlen)
        self.last_delay = deque(maxlen=maxlen)
        self.last_task_obj_num = deque(maxlen=maxlen)
        self.last_task_obj_size = deque(maxlen=maxlen)
        
        self.tasks = deque(maxlen=maxlen)

        for _ in range(maxlen):
            self.cpu_local.append(0)
            self.cpu_other.append(0)
            self.bandwidth_edge_local.append(0)
            self.bandwidth_edge_other.append(0)
            self.last_decision.append(0)
            self.last_delay.append(0)
            self.last_task_obj_num.append(0)  
            self.last_task_obj_size.append(0)  

    def get_state_vector(self):
        """将 deque 数据转换为 2D NumPy 数组"""
        return np.array([
            #list(self.cpu_local),
            #list(self.cpu_other),
            list(self.bandwidth_edge_local),
            list(self.bandwidth_edge_other),
            #list(self.last_decision),
            #list(self.last_delay),
            #list(self.last_task_obj_num),
            #list(self.last_task_obj_size)
        ])
    
    def get_reward_info(self):
        tasks_eval_list = []

        for task in self.tasks:
            task_id = task.get_task_id()
            task_total_time = task.calculate_total_time()
            task_cloud_edge_transmit_time = task.calculate_cloud_edge_transmit_time()

            tasks_eval_list.append({
                "task_id": task_id,
                "total_time": task_total_time,
                "cloud_edge_transmit_time": task_cloud_edge_transmit_time
            })

        # 按 task_id 从大到小排序
        sorted_tasks_eval_list = sorted(tasks_eval_list, key=lambda x: x["task_id"], reverse=True)

        return sorted_tasks_eval_list


class CloudEdgeEnv():
    def __init__(self, device_info=None, cloud_device=None):
        self.device_info = device_info
        self.device_list = list(self.device_info.values()) + [cloud_device]  #按yaml顺序
        self.local_edge = self.device_list[0]
        self.other_edges = self.device_list[1:-1]
        self.device_info['cloud'] = cloud_device

        self.train_parameters = None
        self.selected_device = [self.local_edge, self.local_edge]      

        self.task_count = 0
        self.max_count = 30
        self.maxlen = 3

        self.state_buffer = StateBuffer(maxlen=self.maxlen)

    def reset(self):
        return self.state_buffer.get_state_vector()

    def step(self, action):   
        selected_edge = random.choice(self.other_edges)
        self.selected_action = action
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
        
        #做出决策后等待一段时间
        time.sleep(1)

        new_state = self.get_new_state() 

        reward = self.compute_reward()

        done = self.check_done()   

        return new_state, reward, done, {}  

    def set_local_edge(self, local_edge):
        self.local_edge = local_edge

    def get_selected_device(self):
        return self.selected_device
    
    def set_train_parameters(self, train_parameters):
        self.train_parameters = train_parameters

    def get_new_state(self):
        return self.state_buffer.get_state_vector()

    def extract_cpu_state(self, resource_table):
        local_edge_cpu = None
        other_edge_cpu = None

        for key, val in resource_table.items():
            if key.startswith('edge'):
                if 'cpu' in val:
                    if key == self.local_edge:
                        local_edge_cpu = val['cpu']
                    else:
                        other_edge_cpu = val['cpu']

        return local_edge_cpu, other_edge_cpu
 
    def extract_bandwidth_state(self, resource_table):
        bandwidth_value = None

        for key, val in resource_table.items():
            if 'bandwidth' in val:
                bandwidth_value = val['bandwidth']
                if key.startswith('edge'):  # 如果是 edge，立即返回
                    break

        return bandwidth_value, bandwidth_value  #same
    
    def check_done(self):
        self.task_count += 1
        done = self.task_count >= self.max_count
        if done:
            self.task_count = 0
        return done

    def compute_reward(self):
        sorted_tasks_eval_list = self.state_buffer.get_reward_info()

        n = 3
        top_n_tasks = sorted_tasks_eval_list[:n]  
        total_time_sum = sum([task["total_time"] for task in top_n_tasks])
        avg_total_time = total_time_sum / n if n > 0 else 0

        print(top_n_tasks[0]['task_id'])
        print(top_n_tasks[1]['task_id'])
        print(top_n_tasks[2]['task_id'])

        reward = -avg_total_time + 1

        return reward


    def update_resource_state(self, resource_table):
        local_edge_cpu, other_edge_cpu = self.extract_cpu_state(resource_table)
        bandwidth_local, bandwidth_other = self.extract_bandwidth_state(resource_table)

        if local_edge_cpu is not None:
            self.update_cpu_local(local_edge_cpu)
        if other_edge_cpu is not None:
            self.update_cpu_other(other_edge_cpu)
        
        if bandwidth_local is not None:
            self.update_bandwidth_local(bandwidth_local)
        if bandwidth_other is not None:
            self.update_bandwidth_other(bandwidth_other)

    def update_cpu_local(self, value):
        value /= 100
        self.state_buffer.cpu_local.append(value)

    def update_cpu_other(self, value):
        value /= 100
        self.state_buffer.cpu_other.append(value)

    def update_bandwidth_local(self, value):
        value /= 100
        self.state_buffer.bandwidth_edge_local.append(value)

    def update_bandwidth_other(self, value):
        value /= 100
        self.state_buffer.bandwidth_edge_other.append(value)

    def update_decision(self, value):
        value /= 5
        self.state_buffer.last_decision.append(value)

    def update_delay(self, value):
        self.state_buffer.last_delay.append(value)

    def update_task_obj_num(self, value):
        if isinstance(value, list) and len(value) > 0:
            avg = sum(value) / len(value)
        else:
            avg = 0

        avg /= 10
        self.state_buffer.last_task_obj_num.append(avg)

    def update_task_obj_size(self, value):
        if isinstance(value, list) and len(value) > 0:
            avg = sum(value) / len(value)
        else:
            avg = 0
        self.state_buffer.last_task_obj_size.append(avg)

    def update_tasks(self, task):
        self.state_buffer.tasks.append(task)

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

    state_dim = 2
    history_length = 3
    action_dim = 6

    agent = PPO(state_dim, hidden_dim, history_length, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

    rl_utils.train_on_policy_agent_CloudEdge(env, agent, num_episodes)


