import time
from vuer import Vuer
from vuer.schemas import Hands
from multiprocessing import Array, Process, context
import numpy as np
import asyncio

Value = context._default_context.Value


class TeleVision:
    def __init__(self, cert_file="cert.pem", key_file="key.pem", ngrok=False):
        
        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', port=8000, queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', port=8000,cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)

        # ✅ 启动显示手的协程
        self.vuer.spawn(start=False)(self.show_hands_loop)

        # 创建共享数据
        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except:
            pass

    # ✅ 新增：循环显示手部模型
    async def show_hands_loop(self, session, fps=60):
        session.upsert @ Hands(
            fps=fps,
            stream=True,
            key="hands",
            showLeft=True,   # ✅ 开启左手
            showRight=True   # ✅ 开启右手
        )
        while True:
            await asyncio.sleep(1.0)  # 保持会话激活

    # 数据读取属性
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

    # 启动 TeleVision
    tv = TeleVision(ngrok=False, cert_file="cert.pem", key_file="key.pem")
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    while True:
        time.sleep(0.03)
