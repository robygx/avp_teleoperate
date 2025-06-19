import time
import asyncio
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory
import numpy as np

from multiprocessing import context
Value = context._default_context.Value
np.set_printoptions(suppress=True, precision=4)  # 可选设置精度为 4 位小数

class TeleVision:
    def __init__(self, cert_file="cert.pem", key_file="key.pem", ngrok=False):

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', port=8000, queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', port=8000,cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)


        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        # 注册 spawn 协程：只接收手部数据
        self.vuer.spawn(start=False)(self.spawn_hand_only)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()
    
    def vuer_run(self):
        self.vuer.run()

    async def spawn_hand_only(self, session, fps=60):
        # 👇 这一步是关键：告诉 XR 页面“我需要手势流数据”
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)

        while True:
            # 不做任何图像更新，仅维持连接与 upsert 状态
            await asyncio.sleep(1.0)  # 随便设个慢一点的节奏，防止死循环卡线程

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
            # print("HEAD MATRIX:\n", np.array(self.head_matrix))

        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            # print("[DEBUG] HAND_MOVE received!")  # 放在最前面
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()

                # ✅ 加入打印调试信息
            # print("LEFT HAND MATRIX:\n", np.array(self.left_hand).round(3))
            # print("RIGHT HAND MATRIX:\n", np.array(self.right_hand).round(3))
            
        except: 
            pass


    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
    
if __name__ == '__main__':
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import threading


    # television
    tv = TeleVision(ngrok=False, cert_file="cert.pem", key_file="key.pem")
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    while True:
        time.sleep(0.03)
