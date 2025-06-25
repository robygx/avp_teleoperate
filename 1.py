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
    model = mj.MjModel.from_xml_path("/home/sun/avp_teleoperate/assets/Atom01_urdf/mjcf/atom01.xml")  # 模型XML文件路径:contentReference[oaicite:3]{index=3}
    data = mj.MjData(model)  # 仿真数据对象，用于存储仿真状态:contentReference[oaicite:4]{index=4}
    

    with viewer.launch_passive(model, data) as viewer_inst:
        try:
            user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
            if user_input.lower() == 'r':
                running = True
                counter = 0  # 计数器初始化
                while running:
                    start_time = time.time()
                    head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                    # solve ik using motor data and wrist pose, then use ik results to control arms.
                    time_ik_start = time.time()

                    sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist)
                    time_ik_end = time.time()

                    # 应用 IK 结果到模拟器
                    full_qpos = data.qpos.copy()
                    # IK结果对应MuJoCo模型中手臂10个关节的索引
                    ik_joint_indices = [20, 21, 22, 23, 24, 25, 26, 27, 28 ,29]
                    for i, joint_id in enumerate(ik_joint_indices):
                        full_qpos[joint_id] = sol_q[i]

                    data.qpos[:] = full_qpos
                    print("手臂角度更新后qpos段：", data.qpos[19:29])
                    # 3. 可选：设置初始速度为0
                    data.qvel[:] = 0
                    mj.mj_forward(model, data)

                    # 4. 注入前馈力矩（按 actuator 顺序）
                    arm_actuator_start = model.nu - 10
                    data.ctrl[arm_actuator_start:] = sol_tauff

                    # 5. 推进仿真
                    mj.mj_step(model, data)
                    viewer_inst.sync()




                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                    current_time = time.time()
                    time_elapsed = current_time - start_time
                    sleep_time = max(0, (1 / 30) - time_elapsed)
                    time.sleep(sleep_time)


        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting program...")
        finally:

            print("Finally, exiting program...")
        exit(0)