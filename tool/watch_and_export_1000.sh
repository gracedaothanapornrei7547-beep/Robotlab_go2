#!/usr/bin/env bash
set -euo pipefail

readonly CKPT_DIR="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_flat/2026-10-16_15-08-17"
readonly TARGET_CKPT="${CKPT_DIR}/model_1000.pt"
readonly PYTHON_EXEC="/home/rum/env_isaaclab/bin/python"
readonly EXPORT_SCRIPT="/home/rum/robot_lab/tool/export_policy.py"
readonly EXPORT_DIR="${CKPT_DIR}/exported"

echo "⏳ 正在监听 15-08-17 训练批次的第 1000 轮检查点..."
while true; do
    if [[ -f "${TARGET_CKPT}" ]]; then
        size=$(stat -c%s "${TARGET_CKPT}" 2>/dev/null || echo 0)
        if [[ ${size} -gt 4000000 ]]; then
            echo "🎯 检测到 model_1000.pt 已成功保存！"
            sleep 2
            break
        fi
    fi
    sleep 3
done

echo "📦 自动导出 model_1000.pt 至 TorchScript JIT 和 ONNX..."
"${PYTHON_EXEC}" "${EXPORT_SCRIPT}" --checkpoint "${TARGET_CKPT}" --output_dir "${EXPORT_DIR}"

# 自动刷新 run_mujoco_cpp.sh 的默认模型路径
sed -i "s|readonly MODEL_PATH=.*|readonly MODEL_PATH=\"\${2:-${EXPORT_DIR}/policy.pt}\"|" /home/rum/robot_lab/tool/run_mujoco_cpp.sh

echo "=============================================================================="
echo "✅ 第 1000 轮大步幅 Trot 策略已全部导出并绑定至 MuJoCo 部署脚本！"
echo "  - 模型文件: ${EXPORT_DIR}/policy.pt"
echo "=============================================================================="

