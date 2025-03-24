from collections import deque
import numpy as np

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

    def update(self, cpu_local, cpu_other, bandwidth_local, bandwidth_other, decision, delay, task_num, task):
        """更新状态"""
        self.cpu_local.append(cpu_local)
        self.cpu_other.append(cpu_other)
        self.bandwidth_edge_local.append(bandwidth_local)
        self.bandwidth_edge_other.append(bandwidth_other)
        self.last_decision.append(decision)
        self.last_delay.append(delay)
        self.last_task_obj_num.append(task_num)  
        self.last_task_obj_size.append(task)  

    def get_state_vector(self):
        """将 deque 数据转换为 1D NumPy 数组"""
        return np.concatenate([
            np.array(self.cpu_local),
            np.array(self.cpu_other),
            np.array(self.bandwidth_edge_local),
            np.array(self.bandwidth_edge_other),
            np.array(self.last_decision),
            np.array(self.last_delay),
            np.array(self.last_task_obj_num),
            np.array(self.last_task_obj_size) 
        ])

    def get_state_shape(self):
        """返回状态向量的形状"""
        return self.get_state_vector().shape
