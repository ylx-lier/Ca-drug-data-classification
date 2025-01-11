# classification.py
import xgboost as xgb
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    def train(self, X_train, y_train):
        """Trains the XGBoost model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    def save_model(self, path):
        """Saves the trained model to a file."""
        self.model.save_model(path)