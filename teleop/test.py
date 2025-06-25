import numpy as np
import time
import cv2
import threading
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import mujoco as mj
from mujoco import viewer
from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_ik import Atom_23_ArmIK

# 控制参数
Kp = 200.0
Kd = 2.0
sim_freq = 500      # MuJoCo 仿真频率
ctrl_freq = 30      # 控制器更新频率

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
    while running:
        with ctrl_lock:
            sol_q = shared_ctrl["sol_q"].copy()
            sol_tauff = shared_ctrl["sol_tauff"].copy()

        q_real = data.qpos[arm_joint_indices].copy()
        dq_real = data.qvel[arm_joint_indices].copy()

        tau_pd = Kp * (sol_q - q_real) + Kd * (0 - dq_real)
        # tau_total = sol_tauff + tau_pd
        tau_total = tau_pd

        for i, act_id in enumerate(arm_actuator_indices):
            data.ctrl[act_id] = tau_total[i]

        # # 每 0.5 秒打印一次
        # if step_counter % print_interval_steps == 0:
        #     print("------ Control Debug Info ------")
        #     print("tau_pd:", np.round(tau_pd, 3))
        #     print("sol_tauff:", np.round(sol_tauff, 3))

        step_counter += 1
        mj.mj_step(model, data)
        viewer_inst.sync()
        time.sleep(1.0 / sim_freq)

if __name__ == '__main__':
    tv_wrapper = TeleVisionWrapper()
    arm_ik = Atom_23_ArmIK(Unit_Test = False, Visualization = True)
    # 初始化模型与数据
    model_path = "/home/sun/avp_teleoperate/assets/Atom01_urdf/mjcf/atom02.xml"
    model = mj.MjModel.from_xml_path(model_path)
    data = mj.MjData(model)

    arm_joint_indices = list(range(10))
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
                running = False
                ctrl_thread.join()
                print("Program exited cleanly.")