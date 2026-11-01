#!/usr/bin/env bash
# ==============================================================================
# 自动监听并在第 1000 轮精准停止训练与导出的守护脚本
# 规范约定：严格存放于 tool/ 目录
# ==============================================================================
set -euo pipefail

readonly CKPT_DIR="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_flat/2026-10-16_14-37-49"
readonly TARGET_CKPT="${CKPT_DIR}/model_1000.pt"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly EXPORT_SCRIPT="/home/rum/robot_lab/tool/export_policy.py"
readonly EXPORT_DIR="${CKPT_DIR}/exported"

echo "⏳ 开始监听第 1000 轮检查点: ${TARGET_CKPT} ..."
while true; do
    if [[ -f "${TARGET_CKPT}" ]]; then
        size=$(stat -c%s "${TARGET_CKPT}" 2>/dev/null || echo 0)
        if [[ ${size} -gt 4000000 ]]; then
            echo "🎯 检测到 model_1000.pt 已成功保存 (大小: ${size} 字节)！"
            break
        fi
    fi
    sleep 3
done

echo "🛑 到达 1000 轮，立即停止训练任务..."
pkill -f "scripts/reinforcement_learning/rsl_rl/train.py" || true
sleep 2

echo "📦 自动导出 model_1000.pt 至 TorchScript JIT 和 ONNX..."
"${PYTHON_EXEC}" "${EXPORT_SCRIPT}" --checkpoint "${TARGET_CKPT}" --output_dir "${EXPORT_DIR}"

# 自动刷新 run_mujoco_cpp.sh 的默认模型路径
sed -i "s|readonly MODEL_PATH=.*|readonly MODEL_PATH=\"\${2:-${EXPORT_DIR}/policy.pt}\"|" /home/rum/robot_lab/tool/run_mujoco_cpp.sh

echo "=============================================================================="
echo "✅ 第 1000 轮模型训练已精准截停并自动导出！"
echo "  - JIT 模型: ${EXPORT_DIR}/policy.pt"
echo "  - ONNX 模型: ${EXPORT_DIR}/policy.onnx"
echo "=============================================================================="

