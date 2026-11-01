#!/usr/bin/env bash
# ==============================================================================
# Go2 全向 MoE-CTS 真实物理 MuJoCo C++ 部署引擎启动脚本 (LibTorch + MuJoCo C API)
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

readonly BINARY="/home/rum/robot_lab/tool/deploy_mujoco_cpp/build/mujoco_deploy_go2"
readonly VX="${1:-0.8}"
readonly MODEL_PATH="${2:-/home/rum/go2_rl_robotlab/logs/rsl_rl/go2_moe_cts/2026-10-22_19-20-12_fast_trot_scratch_resumed/exported_1500/policy.pt}"
readonly VY="${3:-0.0}"
readonly WZ="${4:-0.0}"
readonly SCENE_PATH="${5:-/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml}"
readonly TIME_SCALE="${6:-1.0}"

if [[ ! -x "${BINARY}" ]]; then
    echo "[ERROR] 可执行文件不存在，请先编译: ${BINARY}" >&2
    exit 2
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[ERROR] 策略模型文件不存在: ${MODEL_PATH}" >&2
    exit 3
fi

export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/home/rum/build/mujoco/build/lib:/home/rum/env_isaaclab/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "=============================================================================="
echo "⚡ 启动 Go2 全向 MoE-CTS 纯 C++ MuJoCo 部署与交互推演"
echo "  - 二进制: ${BINARY}"
echo "  - 模型路径: ${MODEL_PATH}"
echo "  - 初始指令: vx=${VX} m/s, vy=${VY} m/s, wz=${WZ} rad/s"
echo "  - 显示器: DISPLAY=${DISPLAY}"
echo "=============================================================================="

exec "${BINARY}" "${VX}" "${MODEL_PATH}" "${VY}" "${WZ}" "${SCENE_PATH}" "${TIME_SCALE}"
