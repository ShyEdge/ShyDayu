import threading
import gym  # type: ignore 
#from gym import spaces  # type: ignore
import torch # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np   # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import rl_utils 


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
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数




class CloudEdgeEnv(gym.Env):
    def __init__(self, device_info=None, cloud_device=None):
        super(CloudEdgeEnv, self).__init__()

        self.device_info = device_info
        self.device_info['cloud'] = cloud_device
        self.resource_table = None
        self.device_list = list(self.device_info.keys())  #按顺序

        self.selected_device = cloud_device

        self.condition = threading.Condition()

        self.delay = 0
        self.task_count = 0
        self.max_count = 50
        
        # 状态空间：负载
        self.observation_space = gym.spaces.Box(
            shape=(len(device_info),), dtype=np.float32 
        )

        # 动作空间，目前做的1阶段的
        self.action_space = gym.spaces.Discrete(len(device_info))

    def reset(self):
        new_state = np.full(len(self.device_info), 50, dtype=np.float32)
        return new_state

    


    def step(self, action):  #执行一个动作并返回环境的下一个状态、奖励、是否完成以及附加信息    

        self.selected_device = self.device_list[action]

        with self.condition:  # 进入临界区，确保同步
            # 通知 get_schedule_plan() 设备已选择
            self.condition.notify_all()  # 唤醒等待的线程
            self.condition.wait()  # 阻塞等待条件满足，直到其他线程通知它

        
        reward = -self.delay

        done = self.check_done()    #执行指定数量的task后作为结束标志

        return self.extract_cpu_state(), reward, done, {}

    
        

    def update_resource_table(self, resource_table):   
        self.resource_table = resource_table

    def update_delay(self, delay):
        self.delay = delay


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
        done = self.task_count >= self.max_tasks
        if done:
            self.task_count = 0
        return done
    
    def get_selected_device(self):
        return self.selected_device
    


#--------------------------------------------------------------------------------------------------------------------------------------
# 训练参数
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98  # 奖励折扣
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 使用 CloudEdgeEnv 作为环境
num_edges = 2  # 设定边缘设备数量
env = CloudEdgeEnv(num_edges)

# 获取状态维度和动作维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建 Actor-Critic 智能体
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

# 训练智能体, stop here 
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)



#--------------------------------------------------------------------------------------------------------------------------------------
# 绘制训练曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on CloudEdgeEnv')
plt.show()

# 计算平滑曲线
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on CloudEdgeEnv (Smoothed)')
plt.show()