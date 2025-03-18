import threading
#import gym  # type: ignore 
#from gym import spaces  # type: ignore
import torch # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np   # type: ignore
import random
from . import rl_utils 
from collections import deque


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


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr) 
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

        #print("*****************************************************env update*****************************************************")

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

        self.selected_device = [cloud_device, cloud_device]

        self.condition = threading.Condition()

        self.delay = 0
        self.task_count = 0
        self.max_count = 50

        self.reward_list = []
        self.reward_list_avg = []
        self.delay_list = []
        self.delay_list_avg = []

        self.state_buffer = deque([(0, 0.6)] * 5, maxlen=5)
        
        # 状态空间：负载
        self.observation_space_shape = 10

        #两阶段只有6种可能
        self.action_space_n = 6  


    def reset(self):
        return self.get_state_buffer()


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
        
        reward = self.compute_reward(self.delay)

        self.state_buffer.append((action, reward))

        self.display_rewards(self.delay, reward)

        done = self.check_done()    #执行指定数量的task后作为结束标志

        return self.get_state_buffer(), reward, done, {}  


    def update_resource_table(self, resource_table):   
        self.resource_table = resource_table

    def update_delay(self, delay):
        self.delay = delay

    def set_local_edge(self, local_edge):
        self.local_edge = local_edge

    def get_selected_device(self):
        return self.selected_device

    def extract_cpu_state(self):
        # 提取 cloud.kubeedge 的 CPU 负载
        cloud_cpu = self.resource_table.get("cloud.kubeedge", {}).get("cpu", 0)

        # 提取所有 edge 设备，并按 edge 编号排序
        edge_cpus = []
        for key, value in self.resource_table.items():
            if key.startswith("edge"):
                try:
                    edge_num = int(key[4:])  # 提取 edge 设备编号
                    edge_cpus.append((edge_num, value.get("cpu", 0)))
                except ValueError:
                    continue  # 跳过无法解析的 edge 设备名

        # 按编号排序
        edge_cpus.sort()

        # 生成最终的状态向量
        state = [cloud_cpu] + [cpu for _, cpu in edge_cpus]
        return np.array(state, dtype=np.float32)
    
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

    def compute_reward(delay):
        if delay < 0.5:
            return 1.0
        elif 0.5 <= delay < 0.7:
            return 0.5
        elif 0.7 <= delay < 0.9:
            return 0.2
        elif 0.9 <= delay < 1.5:
            return -0.5
        else:
            return -1.5

    def get_state_buffer(self):
        state = np.array(self.state_buffer).flatten()
        return state

def train_actorcritic_on_policy(env):
    # 训练参数
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98  # 奖励折扣
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space_shape
    action_dim = env.action_space_n

    # 创建 Actor-Critic 智能体
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    # 训练智能体 
    rl_utils.train_on_policy_agent_CloudEdge(env, agent, num_episodes)