#!/usr/bin/env python3
"""
tool/extract_foot_telemetry.py
提取 Unitree Go2 在 MuJoCo 中走直线的全量动力学与四足空间拓扑数据：
1. 四足相对于机身中心（base frame）的 (X, Y, Z) 空间轨迹与平均落足点
2. 前后左右腿间距（前后轴距 Wheelbase、左右轮距 Trackwidth）
3. 真实步频、步长、各腿滞空时间、支撑相占比 (Duty factor)
4. 关节角度分布（hip, thigh, calf）与参考默认姿态对比
"""

import os
import sys
sys.path.insert(0, "/home/rum/robot_lab")
import numpy as np
import mujoco

from tool.sim2sim_mujoco import Go2RobotLabMuJoCoSim

def rot_mat_to_quat(R):
    pass

def quat_rotate_inverse(q, v):
    # q is [w, x, y, z]
    # v is [x, y, z]
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (w * uv + uuv)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Go2 MuJoCo telemetry extractor")
    parser.add_argument("--model_dir", type=str, default="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_flat/2026-10-16_15-08-17/exported")
    parser.add_argument("--vx", type=float, default=0.8)
    parser.add_argument("--scene", type=str, default="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml")
    args = parser.parse_args()

    model_dir = args.model_dir
    scene_path = args.scene
    vx = args.vx

    sim = Go2RobotLabMuJoCoSim(scene_path, model_dir)
    sim.reset()

    # 获取四足的 geom id: Go2 Menagerie 命名为 "FL", "FR", "RL", "RR"
    foot_geom_names = ["FL", "FR", "RL", "RR"]
    foot_geom_ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geom_names]
    foot_labels = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_name_map = dict(zip(foot_geom_names, foot_labels))
    foot_id_map = dict(zip(foot_labels, foot_geom_ids))

    dt_sim = sim.model.opt.timestep
    decimation = int(round(0.02 / dt_sim))
    
    sim_time = 0.0
    
    # 记录列表
    history = {
        "time": [],
        "base_pos": [],
        "base_quat": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "foot_pos_world": {name: [] for name in foot_labels},
        "foot_pos_body": {name: [] for name in foot_labels},
        "foot_contact": {name: [] for name in foot_labels},
        "joint_pos": [],
        "actions": []
    }

    print(f"🚀 开始在 MuJoCo 中提取 6 秒直线行驶数据 (vx={vx} m/s)...")
    
    # 预热 1 秒消除空投初始扰动
    for step in range(350): # 7 秒
        obs = sim.compute_obs(np.array([vx, 0.0, 0.0]))
        action = sim.forward_policy(obs)
        q_target_rl = sim.default_joint_pos_robotlab + sim.action_scale * action
        q_target_mj = q_target_rl[sim.robotlab_to_mujoco]

        for _ in range(decimation):
            tau = sim.kp * (q_target_mj - sim.data.qpos[7:19]) - sim.kd * sim.data.qvel[6:18]
            tau = np.clip(tau, -sim.torque_limit, sim.torque_limit)
            sim.data.ctrl[:] = tau
            mujoco.mj_step(sim.model, sim.data)
            sim_time += dt_sim

        # 记录 50Hz 数据
        if sim_time >= 1.0: # 1秒后记录
            base_p = sim.data.qpos[0:3].copy()
            base_q = sim.data.qpos[3:7].copy() # [w, x, y, z]
            
            # 各足接触检测
            contacts = {name: 0 for name in foot_labels}
            for i in range(sim.data.ncon):
                con = sim.data.contact[i]
                for g_id, name in zip(foot_geom_ids, foot_labels):
                    if con.geom1 == g_id or con.geom2 == g_id:
                        contacts[name] = 1

            history["time"].append(sim_time)
            history["base_pos"].append(base_p)
            history["base_quat"].append(base_q)
            history["base_lin_vel"].append(sim.data.qvel[0:3].copy())
            history["base_ang_vel"].append(sim.data.qvel[3:6].copy())
            
            q_rl = sim.data.qpos[7:19][sim.mujoco_to_robotlab]
            history["joint_pos"].append(q_rl.copy())
            history["actions"].append(action.copy())

            for g_id, name in zip(foot_geom_ids, foot_labels):
                foot_w = sim.data.geom_xpos[g_id].copy()
                history["foot_pos_world"][name].append(foot_w)
                # 转换到机身坐标系: p_body = R^T * (p_world - p_base)
                p_rel = foot_w - base_p
                # 共轭四元数 [w, -x, -y, -z]
                q_conj = np.array([base_q[0], -base_q[1], -base_q[2], -base_q[3]])
                p_body = quat_rotate_inverse(q_conj, p_rel)
                history["foot_pos_body"][name].append(p_body)
                history["foot_contact"][name].append(contacts[name])

    # 转为 numpy
    for k in ["base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_pos", "actions"]:
        history[k] = np.array(history[k])
    for name in foot_labels:
        history["foot_pos_world"][name] = np.array(history["foot_pos_world"][name])
        history["foot_pos_body"][name] = np.array(history["foot_pos_body"][name])
        history["foot_contact"][name] = np.array(history["foot_contact"][name])
    history["time"] = np.array(history["time"])

    print("\n" + "="*70)
    print("📊 【MuJoCo 直线运行深度物理诊断报告】")
    print("="*70)

    # 1. 机身速度与姿态
    actual_vx = np.mean(history["base_lin_vel"][:, 0])
    actual_vy = np.mean(history["base_lin_vel"][:, 1])
    actual_wz = np.mean(history["base_ang_vel"][:, 2])
    mean_z = np.mean(history["base_pos"][:, 2])
    print(f"1. 机身状态:")
    print(f"   - 设定 vx: {vx:.2f} m/s | 实际平均 vx: {actual_vx:.3f} m/s (追踪比: {actual_vx/vx*100:.1f}%)")
    print(f"   - 横向漂移 vy: {actual_vy*100:.2f} cm/s | 偏航角速度 wz: {actual_wz:.3f} rad/s")
    print(f"   - 机身平均高度 Z: {mean_z*100:.1f} cm (站立标称约 28~32 cm)")

    # 2. 四足在机身坐标系下的空间位置 (X: 前向, Y: 左侧, Z: 上向，因此落地点通常是负的)
    print(f"\n2. 四足在机身坐标系（Base Frame）下的平均位置 (X前, Y左, Z上):")
    # 默认直立站立时的四足参考位置
    # 在 default_joint_pos 下:
    # FR: [0.19, -0.15, -0.30] 大致
    # FL: [0.19,  0.15, -0.30]
    # RR: [-0.19, -0.15, -0.30]
    # RL: [-0.19,  0.15, -0.30]
    for name in foot_labels:
        p_b = history["foot_pos_body"][name]
        mean_p = np.mean(p_b, axis=0)
        p_x_range = [np.min(p_b[:, 0]), np.max(p_b[:, 0])]
        p_y_range = [np.min(p_b[:, 1]), np.max(p_b[:, 1])]
        p_z_range = [np.min(p_b[:, 2]), np.max(p_b[:, 2])]
        print(f"   - {name:7s}: X_mean={mean_p[0]:+.3f}m [{p_x_range[0]:+.2f}~{p_x_range[1]:+.2f}], "
              f"Y_mean={mean_p[1]:+.3f}m [{p_y_range[0]:+.2f}~{p_y_range[1]:+.2f}], "
              f"Z_mean={mean_p[2]:+.3f}m [{p_z_range[0]:+.2f}~{p_z_range[1]:+.2f}]")

    # 3. 几何拓扑分析（前后轴距、左右轮距）
    # 前足中心与后足中心
    front_x = (np.mean(history["foot_pos_body"]["FL_foot"][:, 0]) + np.mean(history["foot_pos_body"]["FR_foot"][:, 0])) / 2.0
    rear_x = (np.mean(history["foot_pos_body"]["RL_foot"][:, 0]) + np.mean(history["foot_pos_body"]["RR_foot"][:, 0])) / 2.0
    wheelbase = front_x - rear_x

    front_track = np.mean(history["foot_pos_body"]["FL_foot"][:, 1]) - np.mean(history["foot_pos_body"]["FR_foot"][:, 1])
    rear_track = np.mean(history["foot_pos_body"]["RL_foot"][:, 1]) - np.mean(history["foot_pos_body"]["RR_foot"][:, 1])

    print(f"\n3. 四足几何拓扑 (Stance Geometry):")
    print(f"   - 前后有效轴距 (Wheelbase): {wheelbase*100:.1f} cm (Go2 髋关节前后距约 38.6 cm)")
    print(f"   - 前肢轮距 (Front Track): {front_track*100:.1f} cm (Go2 髋关节左右距约 19.0 cm)")
    print(f"   - 后肢轮距 (Rear Track):  {rear_track*100:.1f} cm")
    print(f"   - 前足相对质心 X: {front_x*100:+.1f} cm | 后足相对质心 X: {rear_x*100:+.1f} cm")

    # 4. 步态时域特征（步频、滞空时间、支撑相占比）
    print(f"\n4. 各腿时域动力学特征:")
    for name in foot_labels:
        c = history["foot_contact"][name]
        duty_factor = np.mean(c)
        # 计算每次触地时间与滞空时间
        # 寻找上升沿 (0 -> 1: touchdown) 与 下降沿 (1 -> 0: liftoff)
        diff = np.diff(c)
        td_indices = np.where(diff == 1)[0]
        lo_indices = np.where(diff == -1)[0]
        
        # 估算周期
        if len(td_indices) >= 2:
            cycle_times = np.diff(history["time"][td_indices])
            cadence = 1.0 / np.mean(cycle_times)
            stride_len = actual_vx / cadence
        else:
            cadence = 0
            stride_len = 0

        # 计算平均滞空时间 (liftoff to next touchdown)
        air_times = []
        for lo in lo_indices:
            next_td = td_indices[td_indices > lo]
            if len(next_td) > 0:
                dt_air = history["time"][next_td[0]] - history["time"][lo]
                air_times.append(dt_air)
        avg_air_time = np.mean(air_times) if len(air_times) > 0 else 0

        print(f"   - {name:7s}: 步频={cadence:.2f} Hz | 步幅={stride_len*100:.1f} cm | 滞空时间={avg_air_time*1000:.1f} ms | 支撑相占比={duty_factor*100:.1f}%")

    # 5. 关节角度平均值与动作幅度
    # 关节顺序: FR(hip, thigh, calf), FL(hip, thigh, calf), RR(hip, thigh, calf), RL(hip, thigh, calf)
    joint_names_rl = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]
    mean_joints = np.mean(history["joint_pos"], axis=0)
    std_joints = np.std(history["joint_pos"], axis=0)
    default_q = sim.default_joint_pos_robotlab

    print(f"\n5. 关节角均值与标准差 (对照默认标称姿态):")
    for i in range(4):
        leg = ["FR", "FL", "RR", "RL"][i]
        idx = i * 3
        print(f"   - {leg}: hip={mean_joints[idx]:+.2f} (def {default_q[idx]:+.2f}), "
              f"thigh={mean_joints[idx+1]:+.2f} (def {default_q[idx+1]:+.2f}, std={std_joints[idx+1]:.2f}), "
              f"calf={mean_joints[idx+2]:+.2f} (def {default_q[idx+2]:+.2f}, std={std_joints[idx+2]:.2f})")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
