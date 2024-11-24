import pandas as pd
from typing import List, Tuple
import numpy as np

def basic_data_type_formulation(
    data: pd.DataFrame,
    numerical_cols: List[str],
    ordinal_cols: List[str],
    binary_cols: List[str],
    multiclass_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Basic data formulation rountines: determine feature types, target variable and task type
    - numerical: ['numerical', 'ordinal']
    - categorical: ['binary', 'multiclass']
    - target: target feature
    - task: classification or regression
    """
    
    numerical_features = []
    categorical_features = []
    
    for col in data.columns:
        if col == target_col:
            continue
        
        if col in numerical_cols or col in ordinal_cols:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
            
    # data type conversion
    data[numerical_features] = data[numerical_features].astype(float)
    data[categorical_features] = data[categorical_features].astype('Int64')
    print(categorical_features)
    
    assert target_col not in numerical_features and target_col not in categorical_features, "Target column cannot be numerical or categorical"
    
    data = data[numerical_features + categorical_features + [target_col]]
        

    return data, {
        'numerical_feature': numerical_features,
        'categorical_feature': categorical_features,
        'target': target_col
    }
    
    
def update_data_type_info(data: pd.DataFrame, data_type_info: dict, target_col: str) -> dict:
    """
    Update data type information
    """
    numerical_features = data_type_info['numerical_feature']
    categorical_features = data_type_info['categorical_feature']
    
    new_numerical_features = []
    new_categorical_features = []
    for col in data.columns:
        if col == target_col:
            continue
        
        if col in numerical_features:
            new_numerical_features.append(col)
        else:
            new_categorical_features.append(col)
    
    assert target_col not in new_numerical_features and target_col not in new_categorical_features, "Target column cannot be numerical or categorical"
    data_type_info['numerical_feature'] = new_numerical_features
    data_type_info['categorical_feature'] = new_categorical_features
    
    return data_type_info
    
