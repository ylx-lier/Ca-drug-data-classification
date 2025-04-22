# classification.py
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
class Classifier:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            n_estimators=100,  # 默认值
            learning_rate=0.1  # 默认值
            )

    def train(self, X_train, y_train, use_balanced_weights=True):
        """Trains the XGBoost model."""
        
        param_grid = {
            'max_depth': [3, 5, 7, 9],          # 扩大搜索范围
            'learning_rate': [0.001, 0.01, 0.1],# 更细粒度
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8],     # 新增参数
            'reg_alpha': [0, 0.1, 1],           # L1正则
            'reg_lambda': [0, 0.1, 1]           # L2正则
        }
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='f1',
            cv=3
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        logging.info(f"Best params: {grid_search.best_params_}")
    
        """Trains the XGBoost model with balanced sample weights."""
        if use_balanced_weights:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test,save_path=None):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
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
        plt.savefig(save_path)
        return metrics, y_pred

    def save_model(self, path):
        """Saves the trained model to a file."""
        self.model.save_model(path)