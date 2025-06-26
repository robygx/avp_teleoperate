import numpy as np
import time
import cv2
import threading
import os
import sys

from datetime import datetime

import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import mujoco as mj
from mujoco import viewer
from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_ik import Atom_23_ArmIK

# 控制参数
Kp_default = 40
Kd_default = 0.5

# 定义每个关节的 Kp 和 Kd，先填充默认值
Kp_array = np.full(10, Kp_default)
Kd_array = np.full(10, Kd_default)

# 为 joint_0 和 joint_5 单独设置控制参数
Kp_array[0] = 61
Kd_array[0] = 1.7
Kp_array[5] = 61
Kd_array[5] = 1.7

Kp_array[1] = 48
Kd_array[1] = 1.4
Kp_array[6] = 48
Kd_array[6] = 1.4

# Kp_array[2] = 80.0
# Kd_array[2] = 2.0
# Kp_array[7] = 80.0
# Kd_array[7] = 2.0

Kp_array[3] = 20
Kd_array[3] = 0.7
Kp_array[8] = 20
Kd_array[8] = 0.7

# Kp_array[4] = 3
# Kd_array[4] = 0.5
# Kp_array[9] = 3
# Kd_array[9] = 0.5


sim_freq = 500      # MuJoCo 仿真频率
ctrl_freq = 30      # 控制器更新频率
#绘制图像
q_real_log = []
sol_q_log = []
time_log = []
start_time_global = time.time()
# 共享变量
shared_ctrl = {
    "sol_q": np.zeros(10),
    "sol_tauff": np.zeros(10)
}
ctrl_lock = threading.Lock()
running = False

def controller_thread(shared_ctrl, ctrl_lock):

    while running:
        start_time = time.time()
        _, left_wrist, right_wrist, _, _ = tv_wrapper.get_data()
        sol_q, sol_tauff = arm_ik.solve_ik(left_wrist, right_wrist)

        with ctrl_lock:
            shared_ctrl["sol_q"][:] = sol_q
            shared_ctrl["sol_tauff"][:] = sol_tauff

        elapsed = time.time() - start_time
        time.sleep(max(0, (1.0 / ctrl_freq) - elapsed))

def mujoco_thread(model, data, arm_joint_indices, arm_actuator_indices, shared_ctrl, ctrl_lock, viewer_inst):
    model.opt.timestep = 1.0 / sim_freq
    print_interval_steps = int(sim_freq * 0.5)  # 每0.5秒打印一次
    step_counter = 0
    start_time_global = time.time()

    while running:
        with ctrl_lock:
            sol_q = shared_ctrl["sol_q"].copy()
            sol_tauff = shared_ctrl["sol_tauff"].copy()

        q_real = data.qpos[arm_joint_indices].copy()
        dq_real = data.qvel[arm_joint_indices].copy()

        tau_pd = np.zeros_like(sol_q)
        for i in range(10):
            tau_pd[i] = Kp_array[i] * (sol_q[i] - q_real[i]) + Kd_array[i] * (0-dq_real[i])
        tau_total = sol_tauff + tau_pd
        # tau_total = sol_tauff 
        # tau_total = tau_pd

        for i, act_id in enumerate(arm_actuator_indices):
            data.ctrl[act_id] = tau_total[i]

        # # # 每 0.5 秒打印一次
        # if step_counter % print_interval_steps == 0:
        #     print("------ Control Debug Info ------")
        #     print("tau_pd:", np.round(tau_pd, 3))
        #     print("sol_tauff:", np.round(sol_tauff, 3))

        step_counter += 1

        # 绘制图像
        elapsed_time = time.time() - start_time_global
        q_real_log.append(q_real.copy())
        sol_q_log.append(sol_q.copy())
        time_log.append(elapsed_time)

        mj.mj_step(model, data)
        viewer_inst.sync()
        time.sleep(1.0 / sim_freq)

if __name__ == '__main__':
    tv_wrapper = TeleVisionWrapper()
    arm_ik = Atom_23_ArmIK(Unit_Test = False, Visualization = True)
    # 初始化模型与数据
    model_path = "/home/wsy/avp_teleoperate_mujoco/assets/Atom01_urdf/mjcf/atom02.xml"
    model = mj.MjModel.from_xml_path(model_path)
    data = mj.MjData(model)

    arm_joint_indices = list(range(10))
    #获取joint索引对应的名称
    joint_names = [model.joint(i).name for i in arm_joint_indices]
    arm_actuator_indices = list(range(10))

    with viewer.launch_passive(model, data) as viewer_inst:
        user_input = input("Please enter 'r' to start:\n")
        if user_input.lower() == 'r':
            running = True
            # 启动控制线程
            ctrl_thread = threading.Thread(target=controller_thread, args=(shared_ctrl, ctrl_lock))
            ctrl_thread.start()

            try:
                mujoco_thread(model, data, arm_joint_indices, arm_actuator_indices, shared_ctrl, ctrl_lock, viewer_inst)
            except KeyboardInterrupt:
                print("KeyboardInterrupt, exiting...")
            finally:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 转换为 NumPy 数组
                q_real_log = np.array(q_real_log)
                sol_q_log = np.array(sol_q_log)
                time_log = np.array(time_log)

                for joint_idx in range(10):
                    joint_name = joint_names[joint_idx]
                    plt.figure(figsize=(8, 4))
                    plt.plot(time_log, q_real_log[:, joint_idx], label=f'q_real ({joint_name})', linewidth=2)
                    plt.plot(time_log, sol_q_log[:, joint_idx], label=f'sol_q ({joint_name})', linestyle='--', linewidth=2)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Joint Angle (rad)")
                    plt.title(f"{joint_name} Tracking Curve")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{joint_name}_tracking_{timestamp}.png")
                    plt.close()
                running = False
                ctrl_thread.join()
                print("Program exited cleanly.")