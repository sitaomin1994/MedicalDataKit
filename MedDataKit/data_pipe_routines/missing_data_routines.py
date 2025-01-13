from typing import Tuple
from abc import ABC, abstractmethod
import pandas as pd


def simple_impute(data: pd.Series, impute_strategy: str, is_categorical: bool = False) -> pd.Series:
    if is_categorical:
        if impute_strategy == 'mode':
            data = data.fillna(data.mode().iloc[0])
        elif impute_strategy == 'other':
            if data.dtype == 'object' or data.dtype == 'category':
                data = data.fillna('None')
            else:
                data = data.fillna(-999)
    else:
        if impute_strategy == 'mean':
            data = data.fillna(data.mean())
        elif impute_strategy == 'median':
            data = data.fillna(data.median())
        elif impute_strategy == 'zero':
            data = data.fillna(0)
        elif impute_strategy == 'other':
            data = data.fillna(-999)
    return data

class MissingDataHandler(ABC):
    
    def __init__():
        pass
    
    @abstractmethod
    def handle_missing_data(self, data: pd.DataFrame, categorical_features: list, **kwargs) -> Tuple[pd.DataFrame, dict]:
        pass
    

class BasicMissingDataHandler(MissingDataHandler):
    """
    Basic missing data mitigation routines
    1. if missing ratio > threshold1, drop the feature
    2. if missing ratio < threshold2, drop the row
    3. if missing ratio > threshold1 and < threshold2, impute with mean/median/mode
    """
    
    def __init__(
        self, 
        threshold1_num: float = 0.5, 
        threshold2_num: float = 0.1,
        threshold1_cat: float = 0.5, 
        threshold2_cat: float = 0.1,
        impute_num: str = 'mean',
        impute_cat: str = 'other',
    ):
        self.threshold1_num = threshold1_num
        self.threshold2_num = threshold2_num
        self.threshold1_cat = threshold1_cat
        self.threshold2_cat = threshold2_cat
        self.impute_num = impute_num
        self.impute_cat = impute_cat
        
        assert self.impute_num in ['mean', 'median', 'zero'], "Invalid impute_num"
        assert self.impute_cat in ['other', 'mode'], "Invalid impute_cat"
    
    def handle_missing_data(
        self, data: pd.DataFrame, categorical_features: list
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Handle missing data
        """
        missing_data_info = {}
        for feature in data.columns:
            missing_ratio = data[feature].isnull().sum() / len(data)
            if feature in categorical_features:
                if missing_ratio > self.threshold1_cat:
                    data = data.drop(columns=[feature])
                    missing_data_info[feature] = {
                        'action': 'drop_col', 'missing_ratio': '{:.2f}'.format(missing_ratio), 
                    }
                elif missing_ratio > self.threshold2_cat:
                    data[feature] = simple_impute(data[feature], self.impute_cat, is_categorical=True)
                    missing_data_info[feature] = {
                        'action': 'impute', 'missing_ratio': '{:.2f}'.format(missing_ratio), 
                        'impute_value': self.impute_cat
                    }
                else:
                    missing_data_info[feature] = {'action': 'drop_row', 'missing_ratio': '{:.2f}'.format(missing_ratio)}
            else:
                if missing_ratio > self.threshold1_num:
                    data = data.drop(columns=[feature])
                    missing_data_info[feature] = {
                        'action': 'drop_col', 'missing_ratio': '{:.2f}'.format(missing_ratio), 
                    }
                elif missing_ratio > self.threshold2_num:
                    data[feature] = simple_impute(data[feature], self.impute_num, is_categorical=False)
                    missing_data_info[feature] = {
                        'action': 'impute', 'missing_ratio': '{:.2f}'.format(missing_ratio), 
                        'impute_value': self.impute_num
                    }
                else:
                    missing_data_info[feature] = {'action': 'drop_row', 'missing_ratio': '{:.2f}'.format(missing_ratio)}
        
        # drop rows with missing values for remaining features
        data = data.dropna()
                
        return data, missing_data_info
    
    
    
    
