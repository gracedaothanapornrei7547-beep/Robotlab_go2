#!/usr/bin/env bash
# ==============================================================================
# Go2 崎岖地形 (Rough) 300 轮热启动微调训练脚本 (Resume Training)
# 基于历史检查点 model_13400.pt 进行外科手术式修正后的热启动继训
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

readonly ROBOT_LAB_ROOT="/home/rum/robot_lab"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly TASK_NAME="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0"
readonly NUM_ENVS="2048"
readonly MAX_ITERATIONS="300"
readonly LOAD_RUN="${1:-2026-10-12_22-42-19}"
readonly CHECKPOINT="${2:-model_13400.pt}"

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
echo "🚀 启动 Unitree Go2 崎岖地形热启动继训 (300 轮)"
echo "  - 任务: ${TASK_NAME}"
echo "  - 环境数: ${NUM_ENVS}"
echo "  - 迭代轮数: ${MAX_ITERATIONS}"
echo "  - 基础 Run: ${LOAD_RUN}"
echo "  - 基础权重: ${CHECKPOINT}"
echo "  - 解释器: ${PYTHON_EXEC}"
echo "=============================================================================="

exec env -u DISPLAY "${PYTHON_EXEC}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${TASK_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --resume \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --headless "$@"

