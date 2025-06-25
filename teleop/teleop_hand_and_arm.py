import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
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

if __name__ == '__main__':
    tv_wrapper = TeleVisionWrapper()
    arm_ik = Atom_23_ArmIK(Unit_Test = False, Visualization = True)
    # 加载 Atom 机器人 MJCF 模型，创建 MuJoCo 模型和数据对象
    model = mj.MjModel.from_xml_path("/home/sun/avp_teleoperate/assets/Atom01_urdf/mjcf/atom02.xml")  # 模型XML文件路径:contentReference[oaicite:3]{index=3}
    data = mj.MjData(model)  # 仿真数据对象，用于存储仿真状态:contentReference[oaicite:4]{index=4}
    # 左右手臂关节索引（joint index in qpos/qvel）
    arm_joint_indices = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9 ]  # 共10个
    # actuator 索引（用于 data.ctrl）
    arm_actuator_indices = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9 ] 
    # PD 控制参数
    Kp = 200.0
    Kd = 2.0
    with viewer.launch_passive(model, data) as viewer_inst:
        try:
            user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
            if user_input.lower() == 'r':
                running = True
                # 设置频率参数
                sim_freq = 500          # MuJoCo 仿真频率（单位Hz）
                ctrl_freq = 30          # 控制器更新频率（单位Hz）

                steps_per_ctrl = sim_freq // ctrl_freq  # 每次控制更新前推进的仿真步数
                model.opt.timestep = 1.0 / sim_freq     # 设置 MuJoCo 的仿真步长

                while running:
                    start_time = time.time()
                    head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                    # solve ik using motor data and wrist pose, then use ik results to control arms.
                    time_ik_start = time.time()
                    sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist)
                    time_ik_end = time.time()
                    # print(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                    # 读取当前状态
                    q_real = data.qpos[arm_joint_indices].copy()
                    dq_real = data.qvel[arm_joint_indices].copy()
                    print("sol_q", sol_q)
                    print("q_real", q_real)
                     # PD反馈力矩
                    tau_pd = Kp * (sol_q - q_real) + Kd * (0 - dq_real)
                    # 合成最终力矩
                    tau_total = sol_tauff + tau_pd
                    # 设置控制输入
                    for i, act_id in enumerate(arm_actuator_indices):
                        data.ctrl[act_id] = tau_total[i]
                    mj.mj_step(model, data)
                    # 同步可视化
                    viewer_inst.sync()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                    current_time = time.time()
                    time_elapsed = current_time - start_time
                    print("time_elapsed", time_elapsed)
                    sleep_time = max(0, (1 / ctrl_freq) - time_elapsed)
                    time.sleep(sleep_time)


        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting program...")
        finally:

            print("Finally, exiting program...")
        exit(0)
