#!/usr/bin/env bash
set -euo pipefail

TARGET_FILE="${1:-/home/rum/robot_lab/logs/rsl_rl/unitree_go2_flat/2026-10-16_17-51-56/model_500.pt}"

echo "Waiting for ${TARGET_FILE} to be created..."
while [ ! -f "${TARGET_FILE}" ]; do
    sleep 3
done

# 等待 2 秒确保 PyTorch 完全写入并关闭文件句柄
sleep 2
echo "✅ 目标模型已生成: ${TARGET_FILE} (大小: $(stat -c%s "${TARGET_FILE}" 2>/dev/null || stat -f%z "${TARGET_FILE}" 2>/dev/null) 字节)"

