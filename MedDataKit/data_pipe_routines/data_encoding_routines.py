import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder

def basic_numerical_encoding(
    data: pd.DataFrame,
    numerical_features: List[str],
    standardize: bool = True,
    normalize: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Basic encoding routines for numerical features by simple standardization and normalization
    """
    min_val_pre, max_val_pre = data[numerical_features].min().to_dict(), data[numerical_features].max().to_dict()
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    scaler = MinMaxScaler() if normalize else None
    if scaler is not None:
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    min_val, max_val = data[numerical_features].min().to_dict(), data[numerical_features].max().to_dict()
    
    data_encoding_log = {'standardize': standardize, 'normalize': normalize, 'info': {
        feature: {
            'min_pre': min_val_pre[feature],
            'max_pre': max_val_pre[feature],
            'min_post': min_val[feature],
            'max_post': max_val[feature]
        } for feature in numerical_features
    }}
    
    return data, data_encoding_log


def basic_categorical_encoding(
    data: pd.DataFrame,
    categorical_features: List[str],
    max_categories: int = 10
) -> Tuple[pd.DataFrame, dict]:
    """
    Basic encoding routines for categorical features by one-hot encoding with a max number of 10 categories
    """
    
    dim_pre = len(data.columns)
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore', max_categories=max_categories, drop = 'first'
    )
    encoded_features = encoder.fit_transform(data[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
    data = pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)

    # encoding log
    data_encoding_log = {
        'max_categories': max_categories,
        'cat_dim_pre': len(categorical_features),
        'cat_dim_post': len(encoded_feature_names),
        'dim_pre': dim_pre,
        'dim_post': len(data.columns),
        'encoded_cat_features': encoded_feature_names.tolist()
    }
    
    return data, data_encoding_log
