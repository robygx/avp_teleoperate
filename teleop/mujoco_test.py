import mujoco as mj
from mujoco import viewer



# 加载 Atom 机器人 MJCF 模型，创建 MuJoCo 模型和数据对象
model = mj.MjModel.from_xml_path("path/to/atom_robot.xml")  # 模型XML文件路径:contentReference[oaicite:3]{index=3}
data = mj.MjData(model)  # 仿真数据对象，用于存储仿真状态:contentReference[oaicite:4]{index=4}


# 将关节位置设置为 sol_q 解算结果
data.qpos[:] = sol_q  # 注意：若模型有自由基座，自由度也包含在 qpos 中，需要相应处理
data.qvel[:] = 0      # 将关节速度初始为0（静止状态）
mj.mj_forward(model, data)  # 前向计算一次，使得后续的派生量(如位置、碰撞检测等)更新

data.ctrl[:] = sol_tauff  # 将前馈关节力矩设置为 sol_tauff:contentReference[oaicite:9]{index=9}


mj.mj_step(model, data)  # 推进仿真一步:contentReference[oaicite:10]{index=10}

viewer.launch(model, data)  # 启动交互式可视化界面:contentReference[oaicite:12]{index=12}



viewer.launch(model, data)  # 启动交互式可视化界面:contentReference[oaicite:12]{index=12}