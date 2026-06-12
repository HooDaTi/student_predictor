import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model_strategy import ModelStrategy

class ModelTrainer:
    def __init__(self, strategy: ModelStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelStrategy):
        self.strategy = strategy

    def prepare_data(self, df: pd.DataFrame):
        # Target
        df['passed_math'] = (df['math score'] >= 50).astype(int)
        
        X = df.drop(columns=['math score', 'passed_math'])
        y = df['passed_math']
        
        X = pd.get_dummies(X, drop_first=True)
        
        return X, y

    def run_training(self, df: pd.DataFrame):
        X, y = self.prepare_data(df)

        self.train_columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.strategy.train(X_train, y_train)
        predictions = self.strategy.predict(X_test)

        metrics = {
                'Accuracy': accuracy_score(y_test, predictions),
                'Precision': precision_score(y_test, predictions, zero_division=0),
                'Recall': recall_score(y_test, predictions, zero_division=0),
                'F1-Score': f1_score(y_test, predictions, zero_division=0)
        }
    
        return metrics

    def real_prediction(self, student_data: dict):
        df_real_student = pd.DataFrame([student_data])
        df_real_prepared = pd.get_dummies(df_real_student)

        df_real_prepared = df_real_prepared.reindex(columns=self.train_columns, fill_value=0)
        
        prediction = self.strategy.predict(df_real_prepared)
        return prediction[0]