from typing import Tuple
import pandas as pd

def basic_missing_mitigation(data: pd.DataFrame, threshold1: float = 0.3) -> Tuple[pd.DataFrame, dict]:
    """
    Basic missing data mitigation routines
    1. if missing ratio > threshold, drop the feature
    2. if missing ratio < threshold, drop the row
    3. todo: impute with mean/median/mode (not implemented yet)
    """
    missing_data_log = {'data_dim_pre': data.shape}
    
    for feature in data.columns:
        missing_ratio = data[feature].isnull().sum() / len(data)
        if missing_ratio > threshold1:
            data = data.drop(columns=[feature])
            missing_data_log[feature] = {'action': 'drop_col', 'missing_ratio': '{:.2f}'.format(missing_ratio), 'threshold': threshold1}
        else:
            missing_data_log[feature] = {'action': 'drop_row', 'missing_ratio': '{:.2f}'.format(missing_ratio), 'threshold': threshold1}
    
    data = data.dropna()
    
    missing_data_log['data_dim_post'] = data.shape
    
    return data, missing_data_log
