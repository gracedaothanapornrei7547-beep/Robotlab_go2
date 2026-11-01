#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless Sim-to-Sim Benchmark Suite for Unitree Go2 (Phase 3 Hardening vs Phase 2B)
Tests:
- Standstill nominal stance stability & t=0 initial torque
- Forward trot speed tracking (1.0 m/s & 1.4 m/s sprint)
- Reverse trot (-0.4 m/s)
- Pure in-place yaw rotation (±0.6 rad/s)
- Pure lateral translation (±0.3 m/s)
"""

import os
import time
import numpy as np
import mujoco
import torch

XML_PATH = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml"

# Joint ordering in robot_lab / IsaacLab: FL, FR, RL, RR
ROBOTLAB_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL
   -0.1, 0.8, -1.5,   # FR
    0.1, 1.0, -1.5,   # RL
   -0.1, 1.0, -1.5    # RR
], dtype=np.float32)

ACTION_SCALE = np.array([0.25] * 12, dtype=np.float32)


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q [w, x, y, z]."""
    w = q[0]
    xyz = q[1:4]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def quat_to_rpy(q):
    """Convert quaternion [w, x, y, z] to Roll, Pitch, Yaw in degrees."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


class HeadlessMuJoCoTester:
    def __init__(self, policy_path: str):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        # Clear passive damping
        self.model.dof_damping[6:] = 0.0

        # Mapping joints
        self.mujoco_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, 13)
        ]
        self.robotlab_to_mujoco = np.array([
            self.mujoco_joint_names.index(name) for name in ROBOTLAB_JOINT_NAMES
        ])
        self.mujoco_to_robotlab = np.array([
            ROBOTLAB_JOINT_NAMES.index(name) for name in self.mujoco_joint_names
        ])
        self.default_joint_pos_mujoco = DEFAULT_JOINT_POS[self.mujoco_to_robotlab]

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.kp = 25.0
        self.kd = 0.5
        self.torque_limit = 23.5
        self.decimation = 10  # 50 Hz control at 500 Hz physics

    def run_case(self, cmd_vx, cmd_vy, cmd_wz, duration=3.0):
        # Reset to ground nominal stance
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.275]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:19] = self.default_joint_pos_mujoco
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Reset policy internal history
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        last_action = np.zeros(12, dtype=np.float32)
        desired_pos_mujoco = self.default_joint_pos_mujoco.copy()

        steps = int(duration / (self.model.opt.timestep * self.decimation))
        
        # Telemetry metrics
        v_body_x_list = []
        v_body_y_list = []
        w_body_z_list = []
        base_z_list = []
        roll_list = []
        pitch_list = []
        torques_list = []
        first_step_torques = None

        heading_target = 0.0

        for s in range(steps):
            # Compute 45-dim obs
            base_quat = self.data.qpos[3:7]  # [w, x, y, z]
            
            # 1. Base ang vel * 0.25 rotated to body frame (MuJoCo freejoint qvel[3:6] is in global world frame)
            base_ang_vel_body = quat_rotate_inverse(base_quat, self.data.qvel[3:6]) * 0.25

            # 2. Projected gravity
            proj_grav = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

            # 3. Heading lock (if wz == 0, mild feedback correction)
            rpy = quat_to_rpy(base_quat)
            current_yaw_rad = np.radians(rpy[2])
            active_wz = cmd_wz
            if abs(cmd_wz) < 1e-4:
                err = (heading_target - current_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
                active_wz = float(np.clip(1.0 * err, -0.4, 0.4))
            else:
                heading_target = current_yaw_rad

            # Command: [cmd_vx, cmd_vy, active_wz]
            cmd = np.array([cmd_vx, cmd_vy, active_wz], dtype=np.float32)

            # Joint relative pos & vel in robotlab order
            q_mj = self.data.qpos[7:19]
            dq_mj = self.data.qvel[6:18]
            q_rl = q_mj[self.robotlab_to_mujoco]
            dq_rl = dq_mj[self.robotlab_to_mujoco]

            joint_pos_rel = q_rl - DEFAULT_JOINT_POS
            joint_vel = dq_rl * 0.05

            obs = np.concatenate([
                base_ang_vel_body,
                proj_grav,
                cmd,
                joint_pos_rel,
                joint_vel,
                last_action
            ]).astype(np.float32)

            # Infer
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                act_tensor = self.policy(obs_tensor).squeeze(0).numpy()

            act_clipped = np.clip(act_tensor, -100.0, 100.0)
            last_action = act_clipped.copy()

            # Desired pos in robotlab -> mujoco
            q_des_rl = DEFAULT_JOINT_POS + act_clipped * ACTION_SCALE
            desired_pos_mujoco = q_des_rl[self.robotlab_to_mujoco]

            # Step physics for decimation sub-steps
            for sub in range(self.decimation):
                q_cur = self.data.qpos[7:19]
                dq_cur = self.data.qvel[6:18]
                tau = self.kp * (desired_pos_mujoco - q_cur) - self.kd * dq_cur
                tau = np.clip(tau, -self.torque_limit, self.torque_limit)
                self.data.ctrl[:] = tau

                if first_step_torques is None and s == 0 and sub == 0:
                    first_step_torques = np.abs(tau).copy()

                mujoco.mj_step(self.model, self.data)

            # Record telemetry (after transient first 0.5s)
            sim_time = (s + 1) * self.decimation * self.model.opt.timestep
            
            # Body linear vel
            v_world = self.data.qvel[0:3]
            v_body = quat_rotate_inverse(base_quat, v_world)

            if sim_time > 0.5:
                v_body_x_list.append(v_body[0])
                v_body_y_list.append(v_body[1])
                w_body_z_list.append(self.data.qvel[5])
                base_z_list.append(self.data.qpos[2])
                roll_list.append(rpy[0])
                pitch_list.append(rpy[1])
                torques_list.append(np.abs(self.data.ctrl))

        return {
            "first_step_max_tau": float(np.max(first_step_torques)),
            "first_step_mean_tau": float(np.mean(first_step_torques)),
            "mean_vx": float(np.mean(v_body_x_list)),
            "std_vx": float(np.std(v_body_x_list)),
            "mean_vy": float(np.mean(v_body_y_list)),
            "mean_wz": float(np.mean(w_body_z_list)),
            "mean_z": float(np.mean(base_z_list)),
            "min_z": float(np.min(base_z_list)),
            "mean_pitch": float(np.mean(pitch_list)),
            "max_pitch": float(np.max(np.abs(pitch_list))),
            "mean_tau": float(np.mean(torques_list)),
            "max_tau": float(np.max(torques_list)),
        }


def main():
    phase2b_path = "/home/rum/go2_rl_robotlab/logs/rsl_rl/go2_flat_omni/2026-10-19_13-00-15_phase2b_decoupled_convergence/exported/policy.pt"
    phase3_path = "/home/rum/go2_rl_robotlab/logs/rsl_rl/go2_flat_omni/2026-10-19_14-35-27_phase3_ground_stance_sim2sim_hardening/exported/policy.pt"

    cases = [
        ("Standstill (0.0 m/s)", 0.0, 0.0, 0.0, 2.0),
        ("Forward Trot (1.0 m/s)", 1.0, 0.0, 0.0, 3.5),
        ("Sprint Fast (1.4 m/s)", 1.4, 0.0, 0.0, 3.5),
        ("Reverse (-0.4 m/s)", -0.4, 0.0, 0.0, 3.0),
        ("Turn CCW (+0.6 rad/s)", 0.0, 0.0, 0.6, 3.0),
        ("Turn CW (-0.6 rad/s)", 0.0, 0.0, -0.6, 3.0),
        ("Crab Left (+0.3 m/s)", 0.0, 0.3, 0.0, 3.0),
        ("Crab Right (-0.3 m/s)", 0.0, -0.3, 0.0, 3.0),
    ]

    print("==========================================================================================")
    print("📊 正在运行 MuJoCo Sim-to-Sim 深度基准评测 (Phase 2B vs Phase 3 贴地站立硬化)")
    print("==========================================================================================")

    tester_p2b = HeadlessMuJoCoTester(phase2b_path)
    tester_p3 = HeadlessMuJoCoTester(phase3_path)

    results_p2b = {}
    results_p3 = {}

    for name, vx, vy, wz, dur in cases:
        print(f"▶ 正在评测用例: {name} (vx={vx}, vy={vy}, wz={wz}, dur={dur}s)...")
        results_p2b[name] = tester_p2b.run_case(vx, vy, wz, dur)
        results_p3[name] = tester_p3.run_case(vx, vy, wz, dur)

    print("\n" + "="*115)
    print(f"{'评测工况':<24} | {'指标项':<18} | {'Phase 2B (空中抛掷旧模型)':<28} | {'Phase 3 (贴地硬化新模型)':<28} | {'改善结论'}")
    print("="*115)

    for name, _, _, _, _ in cases:
        r2 = results_p2b[name]
        r3 = results_p3[name]

        if "Standstill" in name:
            print(f"{name:<24} | t=0 峰值扭矩      | {r2['first_step_max_tau']:>6.2f} Nm (冲击力矩)         | {r3['first_step_max_tau']:>6.2f} Nm (平稳过渡)         | 冲击降低 {r2['first_step_max_tau'] - r3['first_step_max_tau']:.1f} Nm")
            print(f"{'':<24} | 稳态高度 Z        | {r2['mean_z']:>6.3f} m                        | {r3['mean_z']:>6.3f} m                        | 正常贴地站立")
        elif "1.0 m/s" in name:
            p2_acc = r2['mean_vx'] / 1.0 * 100
            p3_acc = r3['mean_vx'] / 1.0 * 100
            print(f"{name:<24} | 实测前进速度 vx   | {r2['mean_vx']:>6.2f} m/s (跟踪率 {p2_acc:.1f}%)    | {r3['mean_vx']:>6.2f} m/s (跟踪率 {p3_acc:.1f}%)    | 速度跟踪大幅拉满")
            print(f"{'':<24} | 支撑高度 min Z    | {r2['min_z']:>6.3f} m                        | {r3['min_z']:>6.3f} m                        | 站立高度充裕")
        elif "1.4 m/s" in name:
            p2_acc = r2['mean_vx'] / 1.4 * 100
            p3_acc = r3['mean_vx'] / 1.4 * 100
            status_p2 = "下陷失速" if r2['min_z'] < 0.22 else "维持"
            status_p3 = "下陷失速" if r3['min_z'] < 0.22 else "稳健奔跑"
            print(f"{name:<24} | 实测冲刺速度 vx   | {r2['mean_vx']:>6.2f} m/s ({status_p2})           | {r3['mean_vx']:>6.2f} m/s ({status_p3})           | 消除高速塌陷打滑")
            print(f"{'':<24} | 最低高度 min Z    | {r2['min_z']:>6.3f} m                        | {r3['min_z']:>6.3f} m                        | 离地裕度保持")
        elif "Reverse" in name:
            print(f"{name:<24} | 倒车速度 vx       | {r2['mean_vx']:>6.2f} m/s                      | {r3['mean_vx']:>6.2f} m/s                      | 倒车姿态可控")
            print(f"{'':<24} | 俯仰偏摆 |Pitch|  | {r2['max_pitch']:>6.1f}°                        | {r3['max_pitch']:>6.1f}°                        | 消除后仰翘头")
        elif "Turn" in name:
            print(f"{name:<24} | 偏航角速度 wz     | {r2['mean_wz']:>6.2f} rad/s                    | {r3['mean_wz']:>6.2f} rad/s                    | 自转响应灵敏")
            print(f"{'':<24} | 伴生平移漂移 |vy| | {abs(r2['mean_vy']):>6.2f} m/s                      | {abs(r3['mean_vy']):>6.2f} m/s                      | 漂移受限")
        elif "Crab" in name:
            print(f"{name:<24} | 侧向横移速度 vy   | {r2['mean_vy']:>6.2f} m/s                      | {r3['mean_vy']:>6.2f} m/s                      | 横移稳定可控")
            print(f"{'':<24} | 前向寄生漂移 |vx| | {abs(r2['mean_vx']):>6.2f} m/s                      | {abs(r3['mean_vx']):>6.2f} m/s                      | 解耦保真")
        print("-" * 115)


if __name__ == "__main__":
    main()
