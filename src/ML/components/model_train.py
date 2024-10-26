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
            "Logistic_Regression": LogisticRegression(**self.config.parms.get("Logistic_Regression", {})),
            "SVM": SVC(**self.config.parms.get("SVM", {})),
            "Decision_Tree": DecisionTreeClassifier(**self.config.parms.get("Decision_Tree", {})),
            "Random_Forest": RandomForestClassifier(**self.config.parms.get("Random_Forest", {})),
        }
    
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, os.path.join(self.config.root_dir, f"{model_name.replace(' ', '_')}.joblib"))


