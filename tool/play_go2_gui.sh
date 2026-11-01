#!/usr/bin/env bash
# ==============================================================================
# Go2 崎岖地形策略 Isaac Lab 图形界面可视化推演脚本
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

readonly ROBOT_LAB_ROOT="/home/rum/robot_lab"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly TASK_NAME="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0"

# 自动定位最新检查点文件，优先使用传入参数，否则自动获取最新运行的检查点
LATEST_CKPT=$(ls -td /home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough/*/model_*.pt 2>/dev/null | head -n 1 || true)
CHECKPOINT="${1:-${LATEST_CKPT:-/home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-10-15_20-07-17/model_299.pt}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] 检查点文件不存在: ${CHECKPOINT}" >&2
    exit 2
fi

cd "${ROBOT_LAB_ROOT}"

export DISPLAY="${DISPLAY:-:0}"
export OMNI_KIT_ACCEPT_EULA="YES"
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "=============================================================================="
echo "🎮 拉起 Isaac Lab GUI 可视化推演"
echo "  - 任务: ${TASK_NAME}"
echo "  - 模型检查点: ${CHECKPOINT}"
echo "  - 显示器: DISPLAY=${DISPLAY}"
echo "=============================================================================="

# 执行可视化，默认展示 4 个环境在不同地形上的实时攀爬与小跑
exec "${PYTHON_EXEC}" scripts/reinforcement_learning/rsl_rl/play.py \
    --task "${TASK_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    --num_envs 4 \
    "${@:2}"

