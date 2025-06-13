import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 初始化 TensorBoard writer
writer = SummaryWriter(log_dir='./')

# 设置数据根目录
root_dir = '/home/featurize/work/ylx/MEA/data/calcium_data_all'

# 用于收集所有数值的列表
all_values = []

# 遍历文件夹和CSV文件
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, header=None)
            values = df.values.flatten()
            all_values.extend(values)

# 转成 numpy array
all_values = np.array(all_values)

# 写入直方图到 TensorBoard（tag 名可以自定义）
writer.add_histogram('./histogram', all_values, global_step=0)

# 关闭 writer
writer.close()

print("Histogram written to TensorBoard. Run `tensorboard --logdir=runs` to view.")
