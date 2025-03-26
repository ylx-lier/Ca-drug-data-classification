import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 加载训练好的XGBoost模型
model = xgb.XGBClassifier()
model.load_model('/home/featurize/work/ylx/MEA/gae/xgboost_model.json')  # 将路径替换为你的模型路径

# 加载测试数据
# 这里假设你有test_embeddings和y_test
# test_embeddings = ...
# y_test = ...

# 使用模型进行预测
y_pred = model.predict(test_embeddings)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 打印分类报告
print("Classification Report:\n", classification_report(y_test, y_pred))
