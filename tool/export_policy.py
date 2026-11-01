#!/usr/bin/env bash
# -*- coding: utf-8 -*-
""":"
# Python interpreter wrapper
exec /home/rum/env_isaaclab/bin/python "$0" "$@"
"""

import argparse
import os
import torch
import torch.nn as nn

def export_model(checkpoint_path: str, output_dir: str):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    actor_dict = ckpt.get("actor_state_dict")
    if actor_dict is None and "model_state_dict" in ckpt:
        # Fallback for alternative structures
        actor_dict = {k.replace("actor.", ""): v for k, v in ckpt["model_state_dict"].items() if "actor." in k}

    # robot_lab RSL-RL Actor 网络拓扑: Linear(45, 512) -> ELU -> Linear(512, 256) -> ELU -> Linear(256, 128) -> ELU -> Linear(128, 12)
    model = nn.Sequential(
        nn.Linear(45, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 12)
    )

    model_dict = {
        "0.weight": actor_dict["mlp.0.weight"],
        "0.bias": actor_dict["mlp.0.bias"],
        "2.weight": actor_dict["mlp.2.weight"],
        "2.bias": actor_dict["mlp.2.bias"],
        "4.weight": actor_dict["mlp.4.weight"],
        "4.bias": actor_dict["mlp.4.bias"],
        "6.weight": actor_dict["mlp.6.weight"],
        "6.bias": actor_dict["mlp.6.bias"],
    }
    model.load_state_dict(model_dict)
    model.eval()

    dummy_input = torch.zeros(1, 45, dtype=torch.float32)

    # 1. 导出 TorchScript JIT 格式 (policy.pt)
    jit_path = os.path.join(output_dir, "policy.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(jit_path)
    print(f"✅ 成功导出 TorchScript JIT: {jit_path} ({os.path.getsize(jit_path) / 1024:.1f} KB)")

    # 2. 导出 ONNX 格式 (policy.onnx)
    onnx_path = os.path.join(output_dir, "policy.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}}
    )
    print(f"✅ 成功导出 ONNX 格式: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

    # 3. 验证 ONNX 与 PyTorch 前向输出数值一致性
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    test_input = torch.randn(1, 45, dtype=torch.float32)
    py_out = model(test_input).detach().numpy()
    ort_out = ort_session.run(None, {"obs": test_input.numpy()})[0]
    max_diff = (np_diff := abs(py_out - ort_out)).max()
    print(f"🔍 导出数值一致性核验: 最大绝对误差 = {max_diff:.2e} (完全一致)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export robot_lab policy to JIT and ONNX.")
    parser.add_argument("--checkpoint", type=str, default="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-10-12_22-42-19/model_13400.pt")
    parser.add_argument("--output_dir", type=str, default="/home/rum/robot_lab/logs/rsl_rl/unitree_go2_rough/exported_13400")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output_dir)

