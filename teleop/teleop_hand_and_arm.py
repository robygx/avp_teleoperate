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

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_ik import Atom_23_ArmIK

if __name__ == '__main__':
    tv_wrapper = TeleVisionWrapper()

    arm_ik = Atom_23_ArmIK(Unit_Test = False, Visualization = True)

    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            running = True
            counter = 0  # 计数器初始化
            while running:
                start_time = time.time()
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                # counter += 1  # 每次循环，计数器加 1
                # 每 10 次打印一次
                # if counter >= 10:
                #     print("head_rmat:\n", head_rmat)
                #     print("left_wrist:\n", left_wrist)
                #     print("right_wrist:\n", right_wrist)                 
                #     # 重置计数器
                #     counter = 0

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()

                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist)
                time_ik_end = time.time()

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