#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim2Sim Deployment Pipeline for Unitree Go2 (robot_lab to MuJoCo)
严格遵循 agent.md 隔离规范，存放于 tool/ 目录。

特性：
- 严格对接 robot_lab 45 维观测空间拓扑与特征缩放 (base_ang_vel 0.25, joint_vel 0.05 等)
- 双向拓扑对齐 (robot_lab: [FR, FL, RR, RL] <-> MuJoCo Menagerie: [FL, FR, RL, RR])
- 真实电机动作缩放 (hip 0.125, thigh/calf 0.25) 与 Go2HV 峰值力矩限制 (23.5 Nm)
- 支持平地 (flat) 与连续台阶 (stairs_5steps, stairs_5steps_5cm, stairs_10cm) 等多种地形
- 支持 JIT (policy.pt) 与 ONNX (policy.onnx) 推理
- 支持 GUI 可视化运行与纯 headless 自动化评测
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

DEFAULT_MODEL_DIR = "/home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-10-31_23-03-20/exported_8000"
DEFAULT_ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"


def quat_apply_inverse(quat, vec):
    """把世界坐标系下的 3D 向量旋转到机身坐标系下 (等价于 IsaacLab 的 quat_apply_inverse)
    quat: [w, x, y, z]
    """
    w = quat[0]
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def quat_to_euler_rpy(quat):
    """四元数 [w, x, y, z] 转欧拉角 Roll, Pitch, Yaw (单位: 度)"""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


class Go2RobotLabMuJoCoSim:
    def __init__(self, scene_path: str, model_dir: str, kp=25.0, kd=0.5, torque_limit=23.5, use_jit=True):
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

        # 1. 加载 MuJoCo 模型
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo 场景 XML 文件未找到: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # 清除 Menagerie 关节自带的被动阻尼，严格对齐 IsaacLab PhysX
        self.model.dof_damping[6:] = 0.0

        # 2. 关节拓扑映射
        # robot_lab 顺序: FR(3), FL(3), RR(3), RL(3)
        self.robotlab_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        # robot_lab 默认角度
        self.default_joint_pos_robotlab = np.array([
            0.0, 0.8, -1.5,   # FR
            0.0, 0.8, -1.5,   # FL
            0.0, 0.8, -1.5,   # RR
            0.0, 0.8, -1.5    # RL
        ], dtype=np.float32)

        # robot_lab 动作缩放: hip 0.125, 其余 0.25
        self.action_scale = np.array([
            0.125, 0.25, 0.25, # FR
            0.125, 0.25, 0.25, # FL
            0.125, 0.25, 0.25, # RR
            0.125, 0.25, 0.25  # RL
        ], dtype=np.float32)

        # MuJoCo 实际关节列表 (1..12)
        self.mujoco_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, 13)
        ]

        # 双向索引置换映射
        self.mujoco_to_robotlab = np.array([self.mujoco_joint_names.index(name) for name in self.robotlab_joint_names])
        self.robotlab_to_mujoco = np.array([self.robotlab_joint_names.index(name) for name in self.mujoco_joint_names])

        self.default_joint_pos_mujoco = self.default_joint_pos_robotlab[self.robotlab_to_mujoco]

        # 3. 加载策略模型 (优先 JIT，其次 ONNX)
        jit_file = os.path.join(model_dir, "policy.pt")
        onnx_file = os.path.join(model_dir, "policy.onnx")

        self.use_jit = use_jit and os.path.exists(jit_file)
        if self.use_jit:
            print(f"🧠 加载 TorchScript JIT 模型: {jit_file}")
            self.policy_jit = torch.jit.load(jit_file, map_location="cpu")
            self.policy_jit.eval()
        elif os.path.exists(onnx_file):
            import onnxruntime as ort
            print(f"🧠 加载 ONNX 模型: {onnx_file}")
            self.ort_session = ort.InferenceSession(onnx_file)
        else:
            raise FileNotFoundError(f"在 {model_dir} 中未找到 policy.pt 或 policy.onnx")

        # 状态缓存
        self.last_action = np.zeros(12, dtype=np.float32)
        self.action_clip = 100.0

    def reset(self, init_pos=None):
        mujoco.mj_resetData(self.model, self.data)
        if init_pos is not None:
            self.data.qpos[0:3] = init_pos
        else:
            self.data.qpos[0:3] = [0.0, 0.0, 0.38]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # [w, x, y, z]
        self.data.qpos[7:19] = self.default_joint_pos_mujoco
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def compute_obs(self, cmd_vel):
        """严格按照 robot_lab 的 45 维观测规范构建输入
        0:3   base_ang_vel * 0.25
        3:6   projected_gravity * 1.0
        6:9   velocity_commands * 1.0
        9:21  joint_pos_rel * 1.0
        21:33 joint_vel * 0.05
        33:45 last_action * 1.0
        """
        base_quat = self.data.qpos[3:7] # [w, x, y, z]

        # 1. 局部坐标系角速度 (MuJoCo freejoint 的 qvel[3:6] 本身即为 body frame 角速度)
        gyro = self.data.sensor("gyro").data if "gyro" in [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)] else self.data.qvel[3:6]
        base_ang_vel = np.array(gyro, dtype=np.float32) * 0.25

        # 2. 投影重力
        projected_gravity = quat_apply_inverse(base_quat, np.array([0.0, 0.0, -1.0])) * 1.0

        # 3. 指令 (闭环航向反馈：严格对齐 IsaacLab heading_command = True 协议，根除走偏)
        cmd_copy = np.array(cmd_vel, dtype=np.float32).copy()
        if abs(cmd_copy[2]) < 1e-4:
            w, x, y, z = base_quat
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            current_yaw = np.arctan2(siny_cosp, cosy_cosp)
            heading_err = (0.0 - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
            cmd_copy[2] = np.clip(1.0 * heading_err, -1.5, 1.5)
        cmd_obs = cmd_copy * 1.0

        # 4. 关节位置与速度 (转换到 robot_lab 顺序)
        q_mujoco = self.data.qpos[7:19]
        dq_mujoco = self.data.qvel[6:18]

        q_robotlab = q_mujoco[self.mujoco_to_robotlab]
        dq_robotlab = dq_mujoco[self.mujoco_to_robotlab]

        joint_pos_rel = (q_robotlab - self.default_joint_pos_robotlab) * 1.0
        joint_vel = dq_robotlab * 0.05

        # 5. 上一拍动作
        actions_obs = self.last_action * 1.0

        # 拼接 45 维观测
        obs_45 = np.concatenate([
            base_ang_vel,
            projected_gravity,
            cmd_obs,
            joint_pos_rel,
            joint_vel,
            actions_obs
        ]).astype(np.float32)

        return np.clip(obs_45, -100.0, 100.0)

    def forward_policy(self, obs):
        if self.use_jit:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                raw_action = self.policy_jit(obs_tensor).squeeze(0).numpy()
        else:
            raw_action = self.ort_session.run(None, {"obs": obs.reshape(1, -1)})[0].flatten()

        action = np.clip(raw_action, -self.action_clip, self.action_clip)
        self.last_action[:] = action
        return action

    def step(self, cmd_vel):
        # 1. 计算 50Hz 策略动作
        obs = self.compute_obs(cmd_vel)
        action_robotlab = self.forward_policy(obs)

        # 2. 动作映射为目标角度
        q_target_robotlab = self.default_joint_pos_robotlab + self.action_scale * action_robotlab
        q_target_mujoco = q_target_robotlab[self.robotlab_to_mujoco]

        # 3. 500Hz 物理执行 (10 个物理子步)
        dt_policy = 0.02
        dt_sim = self.model.opt.timestep
        decimation = int(round(dt_policy / dt_sim))

        for _ in range(decimation):
            q_actual_mujoco = self.data.qpos[7:19]
            dq_actual_mujoco = self.data.qvel[6:18]

            # PD 控制力矩: tau = Kp*(q_des - q) - Kd*dq
            tau = self.kp * (q_target_mujoco - q_actual_mujoco) - self.kd * dq_actual_mujoco
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)

            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        return self.data.qpos[:3], self.data.qpos[3:7]


def run_sim(scene_name: str, model_dir: str, vx=0.5, vy=0.0, wz=0.0, sim_time=15.0, gui=True):
    scene_file_map = {
        "flat": "scene.xml",
        "stairs_5steps": "scene_stairs_5steps.xml",
        "stairs_5steps_5cm": "scene_stairs_5steps_5cm.xml",
        "stairs_10cm": "scene_stairs_10cm.xml",
        "stairs_5steps_15cm": "scene_stairs_5steps_15cm.xml",
    }
    xml_name = scene_file_map.get(scene_name, f"scene_{scene_name}.xml")
    scene_path = os.path.join(DEFAULT_ASSETS_DIR, xml_name)

    sim = Go2RobotLabMuJoCoSim(scene_path, model_dir)
    sim.reset()

    target_cmd = np.array([vx, vy, wz], dtype=np.float32)
    reset_requested = [False]
    base_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")

    print("=" * 70)
    print("🚀 启动 MuJoCo 真实物理 Sim2Sim 推演")
    print(f"  - 场景: {scene_name} ({scene_path})")
    print(f"  - 初始速度指令: vx={vx} m/s, vy={vy} m/s, wz={wz} rad/s")
    print(f"  - 推演模式: {'GUI 键盘交互视窗' if gui else 'Headless 遥测审计'}")
    if gui:
        print("=" * 70)
        print("  🎮 键盘实时遥控键位映射 (与 DreamWaQ / HIMLoco 全局对齐)")
        print("  [ W / ↑ ] 前进加速 (+0.15 m/s)      [ S / ↓ ] 后退减速 (-0.15 m/s)")
        print("  [   A   ] 向左横移 (+0.10 m/s)      [   D   ] 向右横移 (-0.10 m/s)")
        print("  [ Q / ← ] 原地左自转 (+0.20 rad/s)   [ E / → ] 原地右自转 (-0.20 rad/s)")
        print("  [Space/X] 急停刹车 (速度归零)       [   R   ] 复位机器狗回原点")
        print("  [ C / V ] 切换自由漫游 / 自动跟踪视角 (鼠标左键旋转/右键缩放/中键平移)")
    print("=" * 70)

    if gui:
        viewer_ref = [None]

        def teleop_key_callback(key):
            # C / V: 切换自由视角 / 自动跟随视角
            if key in (67, ord('c'), ord('C'), 86, ord('v'), ord('V')):
                if viewer_ref[0] is not None:
                    v = viewer_ref[0]
                    if v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        v.cam.lookat[:] = sim.data.qpos[0:3]
                        print(f"\r[CAMERA] 切换为自由漫游视角 (Free Camera) - 鼠标可自由旋转/缩放/平移   \n", end="", flush=True)
                    else:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        v.cam.trackbodyid = base_body_id
                        print(f"\r[CAMERA] 切换为跟随视角 (Tracking Camera) - 随狗移动，鼠标可360°环绕   \n", end="", flush=True)
                return

            # W / ↑ (前进加速 +0.15 m/s)
            if key in (87, ord('w'), ord('W'), 265):
                target_cmd[0] = float(np.clip(target_cmd[0] + 0.15, -0.6, 1.8))
                print(f"\r[TELEOP] W / ↑ (前进) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # S / ↓ (后退减速 -0.15 m/s)
            elif key in (83, ord('s'), ord('S'), 264):
                target_cmd[0] = float(np.clip(target_cmd[0] - 0.15, -0.6, 1.8))
                print(f"\r[TELEOP] S / ↓ (后退) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # A (左横移 +0.10 m/s)
            elif key in (65, ord('a'), ord('A')):
                target_cmd[1] = float(np.clip(target_cmd[1] + 0.10, -0.6, 0.6))
                print(f"\r[TELEOP] A (左横移) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # D (右横移 -0.10 m/s)
            elif key in (68, ord('d'), ord('D')):
                target_cmd[1] = float(np.clip(target_cmd[1] - 0.10, -0.6, 0.6))
                print(f"\r[TELEOP] D (右横移) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # Q / ← (左自转 +0.20 rad/s)
            elif key in (81, ord('q'), ord('Q'), 263):
                target_cmd[2] = float(np.clip(target_cmd[2] + 0.20, -1.5, 1.5))
                print(f"\r[TELEOP] Q / ← (左自转) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # E / → (右自转 -0.20 rad/s)
            elif key in (69, ord('e'), ord('E'), 262):
                target_cmd[2] = float(np.clip(target_cmd[2] - 0.20, -1.5, 1.5))
                print(f"\r[TELEOP] E / → (右自转) -> Vx: {target_cmd[0]:+.2f} m/s, Vy: {target_cmd[1]:+.2f} m/s, Wz: {target_cmd[2]:+.2f} rad/s    \n", end="", flush=True)
            # Space / X (急停刹车)
            elif key in (32, 88, ord('x'), ord('X')):
                target_cmd[:] = [0.0, 0.0, 0.0]
                print(f"\r[TELEOP] SPACE / X (急停刹车) -> 速度归零!                                             \n", end="", flush=True)
            # R (复位)
            elif key in (82, ord('r'), ord('R')):
                reset_requested[0] = True
                target_cmd[:] = [vx, vy, wz]
                print(f"\r[TELEOP] R (复位) -> 机器狗回到原点!                                                 \n", end="", flush=True)

        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=teleop_key_callback) as viewer:
            viewer_ref[0] = viewer
            # 锁定相机视角至机器狗
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 90.0

            start_t = time.time()
            step_count = 0
            while viewer.is_running():
                step_start = time.time()

                if reset_requested[0]:
                    sim.reset()
                    reset_requested[0] = False
                    time.sleep(0.1)

                pos, quat = sim.step(target_cmd)
                viewer.sync()
                step_count += 1

                # 自动循环重置机制：走过 5.5 米或机身高度异常则重新回起点，方便连续循环观察
                if scene_name.startswith("stairs") and pos[0] > 6.0:
                    print(f"\n🏁 [登顶成功] X={pos[0]:.2f}m，自动重置回起点循环演示...")
                    sim.reset()
                    time.sleep(0.5)
                elif pos[2] < 0.15: # 严重倾覆摔倒重置
                    print(f"\n⚠️ [机身失衡] Z={pos[2]:.2f}m，自动重置...")
                    sim.reset()
                    time.sleep(0.5)

                # 实时状态终端输出
                if step_count % 50 == 0:
                    rpy = quat_to_euler_rpy(quat)
                    print(f"[{time.time()-start_t:5.1f}s] Pos: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m | RPY: ({rpy[0]:.1f}°, {rpy[1]:.1f}°, {rpy[2]:.1f}°) | Cmd: [{target_cmd[0]:.2f}, {target_cmd[1]:.2f}, {target_cmd[2]:.2f}]")

                elapsed = time.time() - step_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)
    else:
        # 纯 Headless 审计模式
        num_steps = int(sim_time / 0.02)
        z_history = []
        for i in range(num_steps):
            pos, quat = sim.step(target_cmd)
            z_history.append(pos[2])
            if i % 100 == 0:
                print(f"[{i*0.02:5.1f}s] Pos: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")

        z_arr = np.array(z_history[50:]) # 丢弃起步稳定期
        heave_p2p = (z_arr.max() - z_arr.min()) * 100.0
        z_std = z_arr.std() * 100.0
        print("-" * 70)
        print("📊 Headless 量化遥测报告:")
        print(f"  - 最终前向移动距离 X: {pos[0]:.2f} m (速度实测: {pos[0]/sim_time:.2f} m/s)")
        print(f"  - 机身垂直振幅峰峰值 (Heave P2P): {heave_p2p:.2f} cm")
        print(f"  - 机身垂直振幅标准差 (Z Std): {z_std:.2f} cm")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 robot_lab MuJoCo Sim2Sim.")
    parser.add_argument("--scene", type=str, default="flat", choices=["flat", "stairs_5steps", "stairs_5steps_5cm", "stairs_10cm", "stairs_5steps_15cm"])
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vx", type=float, default=0.5, help="前向线速度指令 (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="横移线速度指令 (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="偏航角速度指令 (rad/s)")
    parser.add_argument("--time", type=float, default=15.0, help="仿真总时长 (秒)")
    parser.add_argument("--headless", action="store_true", default=False, help="Headless 纯控制台遥测审计模式")
    args = parser.parse_args()

    run_sim(args.scene, args.model_dir, vx=args.vx, vy=args.vy, wz=args.wz, sim_time=args.time, gui=not args.headless)

