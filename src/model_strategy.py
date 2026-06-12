import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod


class ModelStrategy(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class LogisticStrategy(ModelStrategy):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestStrategy(ModelStrategy):
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
