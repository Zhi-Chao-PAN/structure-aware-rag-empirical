#!/bin/bash
set -e

echo "======================================"
echo "    Vision-RAG AutoDL Setup Script    "
echo "======================================"

# 1. Set up environment
echo "[1/4] 安装依赖..."
# Assume standard conda environment provided by AutoDL base image
pip install -r requirements.txt || echo "requirements.txt 警告，尝试直接安装核心依赖"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install colpali-engine>=0.3.0 wandb accelerate pytest pandas pyarrow fastparquet Pillow

# 2. Login wandb (requires WANDB_API_KEY environment variable)
echo "[2/4] 配置 WandB..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️ 未设置 WANDB_API_KEY，将以匿名或离线方式运行。"
else
    wandb login $WANDB_API_KEY
    echo "✅ WandB 登录成功"
fi

# 3. Link AutoDL data directory if exists
echo "[3/4] 检查数据挂载..."
AUTODL_TMP="/root/autodl-tmp/project/data"
LOCAL_DATA="data"
if [ -d "$AUTODL_TMP" ]; then
    echo "发现 AutoDL 数据盘，创建软链接..."
    if [ ! -L "$LOCAL_DATA" ] && [ ! -d "$LOCAL_DATA" ]; then
        ln -s $AUTODL_TMP $LOCAL_DATA
        echo "✅ 数据软链接已创建: $LOCAL_DATA -> $AUTODL_TMP"
    fi
else
    echo "未检测到 AutoDL 数据盘 ($AUTODL_TMP)。假设数据已在项目中。"
fi

# 4. Validate data
echo "[4/4] 验证数据完整性..."
python scripts/validate_data.py

echo "======================================"
echo "✅ 环境部署完成! 可以开始训练。"
echo "建议执行烟雾测试: python train.py --smoke-test --config autodl_train.yaml"
echo "正式训练: accelerate launch train.py --config autodl_train.yaml"
echo "======================================"
