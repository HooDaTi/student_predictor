import pytest
import pandas as pd
from src.model_strategy import LogisticStrategy
from src.trainer import ModelTrainer

@pytest.fixture
def trainer_and_data():
    df = pd.DataFrame({
        "gender": ["female", "male", "female", "male", "female"],
        "math score": [80, 30, 60, 40, 90],
        "reading score": [85, 40, 65, 45, 95],
        "writing score": [88, 35, 60, 42, 92]
    })
    strategy = LogisticStrategy()
    trainer = ModelTrainer(strategy)
    
    return df, trainer

def test_prepare_data(trainer_and_data):
    df, trainer = trainer_and_data
    X, y = trainer.prepare_data(df.copy())

    assert all(val in [0, 1] for val in y)
    assert "math score" not in X.columns
    assert "passed_math" not in X.columns
    assert "reading score" in X.columns
    assert "writing score" in X.columns

def test_run_training_returns_metrics(trainer_and_data):
    df, trainer = trainer_and_data
    metrics = trainer.run_training(df.copy())
    
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-Score" in metrics