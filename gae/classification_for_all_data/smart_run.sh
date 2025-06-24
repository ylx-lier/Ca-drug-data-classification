#!/bin/bash
# 智能运行脚本 - 演示 && 的实际应用

echo "🚀 智能项目启动脚本"
echo ""

# 1. 环境检查
echo "📋 Step 1: 环境检查"
python -c "import torch, numpy, pandas, sklearn; print('✅ 核心依赖已安装')" && \
echo "✅ Python环境检查通过" || {
    echo "❌ 环境检查失败，请安装依赖"
    exit 1
}

echo ""

# 2. 设置权限
echo "🔧 Step 2: 设置权限"
chmod +x main.py && \
chmod +x view_results.py && \
chmod +x start_tensorboard.sh && \
echo "✅ 脚本权限设置完成"

echo ""

# 3. 清理旧结果（可选）
echo "🧹 Step 3: 清理旧结果"
[ -d "runs" ] && echo "发现旧的训练结果" && \
read -p "是否清理旧结果？(y/N): " cleanup && \
[ "$cleanup" = "y" ] && rm -rf runs/* && echo "✅ 清理完成" || echo "🔄 保留旧结果"

echo ""

# 4. 开始训练
echo "🎯 Step 4: 开始训练"
echo "选择模型类型:"
echo "1) simple - 简单图自编码器"
echo "2) graphmae - GraphMAE模型"
echo "3) original - 原始模型"
read -p "请选择 (1-3, 默认1): " model_choice

case $model_choice in
    2) model_type="graphmae" ;;
    3) model_type="original" ;;
    *) model_type="simple" ;;
esac

echo "使用模型: $model_type"
echo "开始训练..." && \
python main.py --model_type $model_type && \
echo "🎉 训练完成！" && \
echo "" && \
echo "🔍 查看结果:" && \
echo "  - 运行 ./view_results.py 查看结果摘要" && \
echo "  - 运行 ./start_tensorboard.sh 启动TensorBoard"
