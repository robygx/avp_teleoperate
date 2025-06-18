import numpy as np
import time
import os
import sys

# 添加工程路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK

if __name__ == '__main__':
    print("初始化 XR 接收器与 IK 模块...")

    # ✅ 初始化 XR → 位姿获取模块
    tv_wrapper = TeleVisionWrapper(binocular=False, img_shape=(720, 1280), img_shm_name="shared_img")

    # ✅ 初始化 IK 模块，开启 MeshCat 可视化
    arm_ik = G1_23_ArmIK(Unit_Test=False, Visualization=True)

    print("系统初始化完成，开始实时读取 XR 数据并在 MeshCat 中仿真 IK 控制...")

    try:
        while True:
            # Step 1: 从 XR 设备获取左右手位姿
            _, left_wrist, right_wrist = tv_wrapper.get_data()

            # Step 2: 使用 IK 模块求解目标关节角，并更新 MeshCat 可视化
            sol_q, _ = arm_ik.solve_ik(left_wrist, right_wrist)

            # 每帧间隔
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n中断运行，程序退出。")
