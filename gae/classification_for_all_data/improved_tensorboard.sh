#!/bin/bash
# improved_tensorboard.sh - 改进的TensorBoard启动脚本

echo "🔍 TensorBoard启动诊断..."

# 检查TensorBoard是否已经在运行
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口6006已被占用，停止现有TensorBoard..."
    pkill -f tensorboard
    sleep 2
fi

# 获取最新的实验目录
RESULTS_DIR="../../results"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 结果目录不存在: $RESULTS_DIR"
    exit 1
fi

LATEST_EXP=$(ls -1 $RESULTS_DIR | grep "^exp" | sort -V | tail -1)

if [ -z "$LATEST_EXP" ]; then
    echo "❌ 没有找到实验结果目录"
    echo "请先运行 python main.py"
    exit 1
fi

TENSORBOARD_DIR="$RESULTS_DIR/$LATEST_EXP/tensorboard"

if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo "❌ TensorBoard目录不存在: $TENSORBOARD_DIR"
    exit 1
fi

# 检查是否有日志文件
LOG_FILES=$(find "$TENSORBOARD_DIR" -name "*.tfevents.*" | wc -l)
if [ "$LOG_FILES" -eq 0 ]; then
    echo "❌ 没有找到TensorBoard日志文件"
    exit 1
fi

echo "✅ 找到 $LOG_FILES 个日志文件"
echo "🚀 启动TensorBoard..."
echo "📁 实验目录: $LATEST_EXP"
echo "📊 TensorBoard目录: $TENSORBOARD_DIR"
echo ""
echo "🌐 访问地址:"
echo "  本地: http://localhost:6006"
echo "  远程: http://$(hostname -I | awk '{print $1}'):6006"
echo ""
echo "按 Ctrl+C 停止TensorBoard"
echo "================================"

# 启动TensorBoard
tensorboard --logdir="$TENSORBOARD_DIR" --port=6006 --host=0.0.0.0 --reload_interval=10
