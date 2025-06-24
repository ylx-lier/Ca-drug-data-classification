# 图神经网络分类实验

## 📊 实验结果可视化

本项目现在支持通过TensorBoard展示各组的分类准确率，让实验结果一目了然！

## 🚀 快速开始

### 1. 运行实验
```bash
python main.py
```

### 2. 查看结果
运行完成后，有两种方式查看结果：

#### 方式一：使用便捷脚本（推荐）
```bash
python view_results.py
```
- 自动显示最新实验的结果摘要
- 询问是否启动TensorBoard
- 一键查看详细结果

#### 方式二：手动启动TensorBoard
```bash
bash start_tensorboard.sh
```
然后在浏览器打开: http://localhost:6006

#### 方式三：直接执行（需要先授权）
```bash
# 首次使用时授权（可选）
chmod +x *.sh *.py

# 然后可以直接执行
./view_results.py
./start_tensorboard.sh
```

## 📈 TensorBoard展示内容

### 1. 分类结果指标
- `Accuracy/{dataset_name}`: 各数据集的准确率
- `Precision/{dataset_name}`: 各数据集的精确率
- `Recall/{dataset_name}`: 各数据集的召回率
- `F1-Score/{dataset_name}`: 各数据集的F1分数

### 2. 整体统计
- `Overall/Average_Accuracy`: 平均准确率
- `Overall/Max_Accuracy`: 最高准确率
- `Overall/Min_Accuracy`: 最低准确率
- `Distribution/Accuracy`: 准确率分布直方图

### 3. 数据信息
- `Sample_Count/{dataset_name}`: 各数据集的样本数量

### 4. 训练过程
- 模型训练的损失曲线

## 📁 结果文件结构

```
results/expN/
├── experiment.log              # 实验日志
├── all_results.json           # 完整结果数据
├── accuracy_comparison.png    # 准确率对比图
├── figures/                   # 可视化图片
│   ├── loss_curve.png        # 训练损失曲线
│   ├── {dataset}_embeddings.png         # 各数据集的嵌入可视化
│   └── {dataset}_confusion_matrix.png   # 各数据集的混淆矩阵
├── models/                    # 保存的模型
│   └── xgboost_model.json
└── tensorboard/              # TensorBoard日志
    ├── classification_results/  # 分类结果
    └── training/               # 训练过程
```

## ⚙️ 模型配置

在 `main.py` 中可以修改模型类型：

```python
model_type = "simple"  # 可选: "simple", "graphmae", "original"
```

- **simple**: 简单图自编码器（推荐，默认）
- **graphmae**: 掩码图自编码器
- **original**: 原始图自编码器

## 🎯 实验结果解读

### 准确率对比
- 在TensorBoard的 `Accuracy` 标签页可以看到各数据集的准确率对比
- 数值越高表示分类效果越好

### 整体性能
- `Overall` 标签页显示所有数据集的统计信息
- 可以了解模型的整体表现

### 数据分布
- `Sample_Count` 显示各数据集的样本数量
- 有助于分析样本不平衡对结果的影响

## 🔧 故障排除

### TensorBoard无法启动
```bash
pip install tensorboard
```

### 端口被占用
修改 `start_tensorboard.sh` 中的端口号：
```bash
tensorboard --logdir="$TENSORBOARD_DIR" --port=6007  # 改为6007
```

### 权限问题
```bash
chmod +x start_tensorboard.sh
chmod +x view_results.py
```

## 📝 注意事项

1. 每次运行实验会创建新的 `expN` 目录
2. TensorBoard日志会累积，定期清理可以节省空间
3. 实验过程中会自动保存所有结果和可视化图片
4. 建议在实验完成后立即查看TensorBoard结果，以便及时发现问题
