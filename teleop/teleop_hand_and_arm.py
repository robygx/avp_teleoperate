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
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')

    args = parser.parse_args()
    print(f"args:{args}\n")

    tv_wrapper = TeleVisionWrapper()

    # arm

    if args.arm == 'G1_23':
        # arm_ik = G1_23_ArmIK()
        arm_ik = G1_23_ArmIK(Unit_Test = False, Visualization = True)

    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            running = True
            counter = 0  # 计数器初始化
            while running:
                start_time = time.time()
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                counter += 1  # 每次循环，计数器加 1
                # 每 10 次打印一次
                if counter >= 10:
                    print("head_rmat:\n", head_rmat)
                    print("left_wrist:\n", left_wrist)
                    print("right_wrist:\n", right_wrist)                 
                    # 重置计数器
                    counter = 0

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()

                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist)
                time_ik_end = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:

        print("Finally, exiting program...")
        exit(0)