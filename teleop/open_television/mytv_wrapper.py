# import numpy as np
# from teleop.open_television.television import TeleVision
# from teleop.open_television.constants import *
# from teleop.utils.mat_tool import mat_update, fast_mat_inv


# class TeleVisionWrapper:
#     def __init__(self, binocular, img_shape, img_shm_name):
#         self.tv = TeleVision(ngrok=False, cert_file="cert.pem", key_file="key.pem")

#     def get_data(self):

#         # --------------------------------wrist-------------------------------------

#         # TeleVision obtains a basis coordinate that is OpenXR Convention
#         head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())
#         left_wrist_vuer_mat, left_wrist_flag  = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())
#         right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())


#         head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
#         left_wrist_mat  = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
#         right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

#         # Change wrist convention: WristMat ((Left Wrist) XR/AppleVisionPro Convention) to UnitreeWristMat((Left Wrist URDF) Unitree Convention)
#         # Reason for right multiply (T_to_unitree_left_wrist) : Rotate 90 degrees counterclockwise about its own x-axis.
#         # Reason for right multiply (T_to_unitree_right_wrist): Rotate 90 degrees clockwise about its own x-axis.
#         unitree_left_wrist = left_wrist_mat @ (T_to_unitree_left_wrist if left_wrist_flag else np.eye(4))
#         unitree_right_wrist = right_wrist_mat @ (T_to_unitree_right_wrist if right_wrist_flag else np.eye(4))

#         # Transfer from WORLD to HEAD coordinate (translation only).
#         unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_mat[0:3, 3]
#         unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_mat[0:3, 3]



#         # --------------------------------offset-------------------------------------

#         head_rmat = head_mat[:3, :3]
#         # The origin of the coordinate for IK Solve is the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
#         # The origin of the coordinate of unitree_left_wrist is HEAD. So it is necessary to translate the origin of unitree_left_wrist from HEAD to WAIST.
#         unitree_left_wrist[0, 3] +=0.15
#         unitree_right_wrist[0,3] +=0.15
#         unitree_left_wrist[2, 3] +=0.45
#         unitree_right_wrist[2,3] +=0.45

#         return head_rmat, unitree_left_wrist, unitree_right_wrist

import numpy as np
import time
from teleop.open_television.television import TeleVision
from teleop.open_television.constants import *
from teleop.utils.mat_tool import mat_update, fast_mat_inv


class TeleVisionWrapper:
    def __init__(self, binocular=False, img_shape=(720, 1280), img_shm_name="shared_img"):
        self.tv = TeleVision(ngrok=False, cert_file="cert.pem", key_file="key.pem")

    def get_data(self):
        head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())
        left_wrist_vuer_mat, left_wrist_flag = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())

        head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

        unitree_left_wrist = left_wrist_mat @ (T_to_unitree_left_wrist if left_wrist_flag else np.eye(4))
        unitree_right_wrist = right_wrist_mat @ (T_to_unitree_right_wrist if right_wrist_flag else np.eye(4))

        unitree_left_wrist[0:3, 3] -= head_mat[0:3, 3]
        unitree_right_wrist[0:3, 3] -= head_mat[0:3, 3]

        unitree_left_wrist[0, 3] += 0.15
        unitree_right_wrist[0, 3] += 0.15
        unitree_left_wrist[2, 3] += 0.45
        unitree_right_wrist[2, 3] += 0.45

        return head_mat[:3, :3], unitree_left_wrist, unitree_right_wrist


# ✅ 添加 main() 以调试 get_data() 结果
if __name__ == '__main__':
    tvw = TeleVisionWrapper()

    print("等待 XR 数据连接中... 请从 XR 浏览器访问正确的 vuer.ai 页面")

    try:
        while True:
            head_rmat, left_wrist, right_wrist = tvw.get_data()

            print("\n[DEBUG] HEAD ROTATION MATRIX:")
            print(np.round(head_rmat, 3))

            print("[DEBUG] LEFT WRIST POSE (Unitree Frame):")
            print(np.round(left_wrist, 3))

            print("[DEBUG] RIGHT WRIST POSE (Unitree Frame):")
            print(np.round(right_wrist, 3))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("中断调试，退出程序。")
