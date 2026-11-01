#!/usr/bin/env bash
# ==============================================================================
# Go2 平地 (Flat) 全速域强化学习训练启动脚本 (0~3.0 m/s 快跑长跑)
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

readonly ROBOT_LAB_ROOT="/home/rum/robot_lab"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly TASK_NAME="RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0"
readonly NUM_ENVS="2048"
readonly MAX_ITERATIONS="${1:-3000}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x "${PYTHON_EXEC}" ]]; then
    echo "[ERROR] Python 解释器不存在: ${PYTHON_EXEC}" >&2
    exit 2
fi

cd "${ROBOT_LAB_ROOT}"

# 基础环境变量与线程池控制
export OMNI_KIT_ACCEPT_EULA="YES"
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================================================="
echo "🚀 启动 Unitree Go2 平地大步幅对角 Trot 步态 (三大约束彻底释放) 强化学习训练"
echo "  - 任务: ${TASK_NAME}"
echo "  - 环境数: ${NUM_ENVS}"
echo "  - 训练轮数: ${MAX_ITERATIONS}"
echo "  - 解释器: ${PYTHON_EXEC}"
echo "=============================================================================="

exec env -u DISPLAY "${PYTHON_EXEC}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${TASK_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless "$@"

