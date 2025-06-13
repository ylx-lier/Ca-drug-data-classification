# classification.py

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np


class Classifier:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100,
            learning_rate=0.1,
            tree_method='auto'

        )

    def train(self, X_train, y_train):
        """Trains the XGBoost model with optional sample balancing."""
        y_train = np.array(y_train).astype(int)
        print("y_train类别分布：", np.unique(y_train, return_counts=True))
        if len(y_train) < 6:
            # 样本太少，直接训练
            self.model.fit(X_train, y_train, verbose=True)
            logging.info("样本太少，跳过超参数搜索，直接训练。")
            return

        # 动态决定交叉验证的折数，防止样本太少出警告
        min_samples_per_class = min([sum(y_train == c) for c in set(y_train)])
        n_splits = min(3, min_samples_per_class) if min_samples_per_class >= 2 else 2
        skf = StratifiedKFold(n_splits=n_splits)

        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='neg_log_loss',  # 多分类对数损失
            cv=skf,
            verbose=1,  # 可以看到搜索进度
            n_jobs=-1   # 全部CPU跑
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        logging.info(f"Best params: {best_params}")

        # 用最优超参数重新初始化模型
        self.model = xgb.XGBClassifier(
            **best_params,
            eval_metric='mlogloss',
            n_estimators=100
        )

        self.model.fit(
            X_train, y_train, 
            
            # eval_set=[(X_train, y_train)],
            verbose=True
        )

    def evaluate(self, X_test, y_test, save_path=None):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(set(y_test)),
            yticklabels=sorted(set(y_test))
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return metrics, y_pred

    def save_model(self, path):
        """Saves the trained model to a file."""
        self.model.save_model(path)
