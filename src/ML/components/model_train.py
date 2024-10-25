from src.ML.entity.config_entity import ModelTrainConfig
import pandas as pd
import os
from src.ML import logger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import joblib

class ModelTrain:

    def __init__(self, config: ModelTrainConfig):
        self.config = config
    
    def train(self):
        train = pd.read_csv(self.config.train_data_path)
        test = pd.read_csv(self.config.test_data_path)

        X_train = train.drop([self.config.target_column], axis=1)
        X_test = test.drop([self.config.target_column], axis=1)
        y_train = train[self.config.target_column]
        y_test = test[self.config.target_column]

        models = {
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
        }
        
        metrics_list = []
        best_model = None
        best_f1_score = 0

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            metrics_list.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model
        
        metrics_df = pd.DataFrame(metrics_list)
        print(metrics_df)

        best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax()]["Model"]
        print(f"\nBest Model: {best_model_name}")
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))


