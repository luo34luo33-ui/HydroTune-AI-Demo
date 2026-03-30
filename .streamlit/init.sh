#!/bin/bash
# Streamlit Cloud 部署初始化脚本
# 此脚本在部署时自动运行

echo "=== 初始化 HydroTune-AI ==="

# 初始化子模块
echo "初始化 Git 子模块..."
git submodule update --init --recursive

# 验证子模块
echo "验证子模块状态..."
for submodule in XAJ-model-structured HBV_model_structured tank-model-structured; do
    if [ -d "$submodule" ] && [ -f "$submodule/main.py" ]; then
        echo "✅ $submodule 已就绪"
    else
        echo "❌ $submodule 未正确初始化"
    fi
done

echo "=== 初始化完成 ==="
