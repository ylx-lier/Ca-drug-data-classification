#!/usr/bin/env python3
"""
最简单的TensorBoard测试
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os

print("🧪 创建最简单的TensorBoard测试...")

# 删除旧的测试目录
test_dir = "simple_test"
if os.path.exists(test_dir):
    import shutil
    shutil.rmtree(test_dir)

# 创建Writer
writer = SummaryWriter(test_dir)

print("📝 写入简单数据...")
# 写入最基本的数据
for i in range(20):
    # 简单的损失曲线
    loss = 1.0 / (i + 1) + 0.1 * np.sin(i)
    writer.add_scalar('Simple/Loss', loss, i)
    
    # 简单的准确率曲线
    acc = 1 - np.exp(-i/10)
    writer.add_scalar('Simple/Accuracy', acc, i)

# 强制刷新
writer.flush()
writer.close()

print(f"✅ 数据写入完成: {test_dir}")

# 检查文件
files = os.listdir(test_dir)
for f in files:
    size = os.path.getsize(os.path.join(test_dir, f))
    print(f"📄 {f}: {size} bytes")

print("\n🚀 启动命令:")
print(f"tensorboard --logdir={test_dir} --port=6010 --host=0.0.0.0")
print("\n🌐 访问地址: http://localhost:6010")
