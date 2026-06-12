import pytest
import pandas as pd
import os
from src.data_handler import DataLoader

@pytest.fixture
def dummy_file():
    test_file = "dummy_test_data.csv"
    
    dummy_data = pd.DataFrame([
        {"gender": "female", "math score": 70},
        {"gender": "male", "math score": 45},
        {"gender": "male", "math score": 45},     
        {"gender": "female", "math score": None}
    ])
    dummy_data.to_csv(test_file, index=False)
    
    yield test_file 
    
    if os.path.exists(test_file):
        os.remove(test_file)

def test_load_data(dummy_file):
    loader = DataLoader(dummy_file)
    df = loader.load_data()
    
    assert len(df) == 4

def test_clean_data(dummy_file):
    loader = DataLoader(dummy_file)
    loader.load_data()
    df = loader.clean_data()
    
    assert len(df) == 2