#!/bin/bash
# start_tensorboard.sh - 启动TensorBoard查看实验结果

# 获取最新的实验目录
RESULTS_DIR="../../results"
LATEST_EXP=$(ls -1 $RESULTS_DIR | grep "^exp" | sort -V | tail -1)

if [ -z "$LATEST_EXP" ]; then
    echo "❌ 没有找到实验结果目录"
    echo "请先运行 python main.py"
    exit 1
fi

TENSORBOARD_DIR="$RESULTS_DIR/$LATEST_EXP/tensorboard"

echo "🚀 启动TensorBoard..."
echo "📁 实验目录: $LATEST_EXP"
echo "📊 TensorBoard目录: $TENSORBOARD_DIR"
echo ""
echo "请在浏览器中打开: http://localhost:6006"
echo "按 Ctrl+C 停止TensorBoard"
echo ""

# 启动TensorBoard
tensorboard --logdir="$TENSORBOARD_DIR" --port=6006 --host=0.0.0.0
