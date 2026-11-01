#!/usr/bin/env bash
# ==============================================================================
# 一键打开 MuJoCo 3D 可视化视窗脚本
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

# 默认场景: flat (平地)
# 支持快捷传参: ./open_mujoco_gui.sh flat 1.0 或 ./open_mujoco_gui.sh stairs_5steps_5cm
SCENE="${1:-flat}"
VX="${2:-0.8}"

export DISPLAY="${DISPLAY:-:0}"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly SCRIPT="/home/rum/isaac/robot_lab/tool/sim2sim_mujoco.py"

echo "=============================================================================="
echo "🚀 打开 MuJoCo 3D 物理引擎可视化视窗"
echo "  - 地形场景: ${SCENE}"
echo "  - 前向速度指令: ${VX} m/s"
echo "  - 显示器: DISPLAY=${DISPLAY}"
echo "=============================================================================="

exec "${PYTHON_EXEC}" "${SCRIPT}" --scene "${SCENE}" --vx "${VX}"

