from actor_critic.network import CloudEdgeEnv, train_actorcritic_on_policy  #这里改为绝对导入了，和实际系统中不符
import random, time
import threading

class ActorCriticAgent():
    def __init__(self, system, agent_id: int, actorcritic_policy: dict = None):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.actorcritic_policy = actorcritic_policy
        self.last_task_delay = 0
        self.train_para = actorcritic_policy['train_parameters']

        self.env = CloudEdgeEnv(actorcritic_policy['device_info'], system.cloud_device)
        

    def get_schedule_plan(self, info):
   
        if self.actorcritic_policy is None:
            return self.actorcritic_policy  #直接返回None

        policy = self.actorcritic_policy.copy()

        #local_device = info['device']
        pipeline = info['pipeline']
        resource_table = info['resource_table']

        cloud_device = self.cloud_device

        #设备信息
        device_info = policy['device_info']
        device_info['cloud'] = cloud_device
        #device_info['local'] = local_device
        
        self.env.update_resource_table(resource_table)  #next_state

        self.env.update_delay(self.last_task_delay)  #reward
        
        # 增加一个同步逻辑，等待选择设备更新
        with self.env.condition:
            # 增加同步逻辑，等待设备更新
            self.env.condition.notify_all()  # 通知 `step` 线程设备已更新
            self.env.condition.wait()  # 阻塞当前线程，等待设备选择

        execute_device = self.env.get_selected_device()
        
        # 修改Pipeline的内容，只针对单阶段
        pipeline = [{**p, 'execute_device': execute_device} for p in pipeline]


        policy.update({'pipeline': pipeline})

        return policy


    def run(self):
        train_actorcritic_on_policy(self.env)

    def update_scenario(self, scenario):
        self.last_task_delay = scenario['delay']


    def continuous_schedule(self):
        cnt = 1
        while True:
            # 模拟实时传入的调度请求信息
            dummy_info = {
                'device': 'local_device',
                'pipeline': [{'task_id': cnt}],
                'resource_table': {
                    "cloud.kubeedge": {"cpu": random.randint(30,70)},
                    "edge1": {"cpu": random.randint(30,70)},
                    "edge2": {"cpu": random.randint(30,70)}
                }
            }
            # 模拟任务延迟信息
            scenario = {'delay': random.uniform(0, 1)}
            time.sleep(scenario['delay'])

            self.update_scenario(scenario)
            # 获取调度计划（此处调用训练后的网络进行决策）
            schedule_plan = self.get_schedule_plan(dummy_info)
            print("Schedule Plan:", schedule_plan)

            cnt += 1


class DummySystem:
    def __init__(self):
        self.cloud_device = "cloud.kubeedge"


dummy_policy = {
    'train_parameters': {
        'dummy_param': True
    },
    'device_info': {
        'edge1': {'cpu': 50},
        'edge2': {'cpu': 50}
    }
}




if __name__ == '__main__':
    # 创建 dummy 系统与 Agent
    system = DummySystem()
    agent = ActorCriticAgent(system, agent_id=1, actorcritic_policy=dummy_policy)

    # 创建线程，设置 daemon=True（主线程结束时子线程自动结束）
    run_thread = threading.Thread(target=agent.run, daemon=True)
    schedule_thread = threading.Thread(target=agent.continuous_schedule, daemon=True)

    # 启动线程
    run_thread.start()
    schedule_thread.start()
    

    # 主线程可以继续做其他事情，此处仅做等待演示
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")





