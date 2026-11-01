#!/usr/bin/env bash
# ==============================================================================
# 训练遥测监控脚本：查看最新日志与启动 TensorBoard
# 规范约定：严格存放于 tool/ 目录，遵循 agent.md 隔离规范
# ==============================================================================
set -euo pipefail

readonly LOG_DIR="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough"

echo "=== Go2 训练日志状态 ==="
if [[ -d "${LOG_DIR}" ]]; then
    echo "最新实验目录列表:"
    ls -lht "${LOG_DIR}" | head -n 5
    echo ""
    LATEST_RUN=$(ls -td "${LOG_DIR}"/*/ 2>/dev/null | head -n 1 || true)
    if [[ -n "${LATEST_RUN}" ]]; then
        echo "最新运行目录: ${LATEST_RUN}"
        echo "检查点文件:"
        ls -lh "${LATEST_RUN}"/model_*.pt 2>/dev/null || echo "（暂无模型检查点，每 100 轮保存一次）"
    fi
else
    echo "日志根目录尚未创建: ${LOG_DIR}"
fi

echo ""
echo "=== TensorBoard 查看指令 ==="
echo "如需在本地查看实时训练曲线，请在终端执行:"
echo "DISPLAY=:0 /home/rum/env_isaaclab/bin/tensorboard --logdir=${LOG_DIR} --port=6006"

