import pandas as pd
from typing import Tuple

def column_check(raw_data: pd.DataFrame) -> dict:
    """
    Check the number of unique values of each column in the raw data
    """
    print("Data shape: ", raw_data.shape)
    column_info = {}
    for col in raw_data.columns:
        missing_rate = raw_data[col].isna().sum() / raw_data.shape[0]
        if raw_data[col].nunique() < 20:
            print(f"{col} ({raw_data[col].dtype} {missing_rate:4.1f}%) => {raw_data[col].nunique()} ({raw_data[col].value_counts().to_dict()})")
        else:
            print(f"{col} ({raw_data[col].dtype} {missing_rate:4.1f}%) => {raw_data[col].nunique()}")

        column_info[col] = {
            "dtype": raw_data[col].dtype,
            "nunique": raw_data[col].nunique(),
            "unique_values": raw_data[col].value_counts().to_dict()
        }

    return column_info


def update_feature_type(
    data: pd.DataFrame, numerical_features: list, categorical_features: list, target_feature: str
) -> Tuple[list, list]:
    """
    Update feature type
    """
    new_numerical_features = []
    new_categorical_features = []
    for col in data.columns:
        if col == target_feature:
            continue
        else:   
            if col in numerical_features:
                new_numerical_features.append(col)
            else:
                new_categorical_features.append(col)
    
    data[new_numerical_features] = data[new_numerical_features].apply(pd.to_numeric, errors='coerce')
    
    return new_numerical_features, new_categorical_features


def handle_targets(
    data: pd.DataFrame, raw_data_config: dict, drop_unused_targets: bool, current_target: str
) -> pd.DataFrame:
    """
    Drop unused target features
    """
    if drop_unused_targets:
        target_features = raw_data_config['target_features']
        for col in target_features:
            if col in data.columns and col != current_target:
                data = data.drop(columns=[col])
    
    return data

