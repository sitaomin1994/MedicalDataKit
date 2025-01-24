from typing import List, Dict, Any, Union, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLTaskPreparationConfig:
    """Configuration for ML task preparation"""
    
    missing_strategy: str = 'impute'                 # strategy for handling missing data
    missing_drop_thres: float = 0.6                  # drop features with missing ratio greater than this threshold
    ordinal_as_numerical: bool = False               # whether to treat ordinal features as numerical features
    categorical_encoding: str = 'ordinal'            # encoding method for categorical features
    numerical_encoding: str = 'standard'             # encoding method for numerical features
    drop_unused_targets: bool = True                # whether to drop unused target features
    #handle_outliers: bool = False
    #outlier_method: str = 'iqr'
    #gaussianize: bool = False
    
    VALID_VALUES = {
        'missing_strategy': ['impute', 'drop', 'keep', 'impute_cat'],
        'categorical_encoding': ['onehot', 'ordinal', 'mixed'],
        'numerical_encoding': ['standard', 'robust', 'quantile', 'yeo-johnson', 'none'],
    }

    VALID_TYPES = {
        'missing_strategy': str,
        'missing_drop_thres': float,
        'ordinal_as_numerical': bool,
        'categorical_encoding': str,
        'numerical_encoding': str,
        'drop_unused_targets': bool,
        #'handle_outliers': bool,
        #'outlier_method': str,
        #'gaussianize': bool
    }

    def __post_init__(self):
        """Validate all attributes after initialization"""
        # Type validation
        for attr, expected_type in self.VALID_TYPES.items():
            value = getattr(self, attr)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Invalid type for {attr}: Expected {expected_type.__name__}, got {type(value).__name__}"
                )
        
        # Value validation for string attributes
        for attr, valid_values in self.VALID_VALUES.items():
            value = getattr(self, attr)
            if value not in valid_values:
                raise ValueError(
                    f"Invalid {attr}: '{value}'. Must be one of {valid_values}"
                )
        
        # Additional validation logic
        if self.missing_drop_thres > 1 or self.missing_drop_thres < 0:
            raise ValueError("missing_drop_thres must be between 0 and 1")
        
        # Additional validation logic
        #if self.handle_outliers and not self.outlier_method:
        #    raise ValueError("outlier_method must be specified when handle_outliers is True")


class MLTaskDataset:
    
    def __init__(
        self,
        data: pd.DataFrame,
        task_name: str,
        target: str,
        task_type: str,
        numerical_features: List[str],
        categorical_features: List[str],
        categorical_encoding: str = 'ordinal',
        numerical_encoding: str = 'standard',
        feature_groups: Dict[str, List[str]] = {},
        feature_mapping: Dict[str, str] = {},
        n_jobs: int = 1,
    ):
        
        # Data
        self.data: pd.DataFrame = data
        
        # Task and Target
        self.task_name = task_name
        self.task_type = task_type
        self.target_feature = target
        self.clf_type = None
        self.num_classes = None
        self.target_codes = None
        
        # Features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_feature_codes = {}
        
        # Data Processing Options
        self.categorical_encoding = categorical_encoding
        self.numerical_encoding = numerical_encoding
        self.n_jobs = n_jobs
        self.one_hot_cols = []
        
        # Federated
        self.fed_cols = []
        self.feature_groups = feature_groups
        self.feature_mapping = feature_mapping
        
        for col, original_col in feature_mapping.items():
            for group_name, features in feature_groups.items():
                if original_col in features:
                    feature_groups[group_name].append(col)
        
        # Validate and Post-process Data
        self._validate_config()
        self._post_processing()
        self._update_feature_groups()
        
        # Missing Data
        self.missing_ratio = None
        self.missing_feature_stats = {}
        self.missing_pattern_stats = {}
        self.missing_data_cleaning_log = {}
        
        # Statistics
        self.feature_stats = {}
    
    @property
    def missing_data_stats(self):
        return {
            'missing_ratio': self.missing_ratio,
            'missing_feature_stats': self.missing_feature_stats,
            'missing_pattern_stats': self.missing_pattern_stats
        }
    
    @property
    def data_processing_config(self):
        return {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'target_feature': self.target_feature,
            'categorical_encoding': self.categorical_encoding,
            'numerical_encoding': self.numerical_encoding,
            'n_jobs': self.n_jobs
        }
        
    @property
    def data_config(self):
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'categorical_feature_codes': self.categorical_feature_codes,
            'target_feature': self.target_feature,
            'task_type': self.task_type,
            'target_codes': self.target_codes,
            'clf_type': self.clf_type,
            'num_classes': self.num_classes,
        }
    
    def _validate_config(self):
        
        assert self.task_type in ['classification', 'regression', 'survival'], "Invalid task type"
        assert (
            self.categorical_encoding in ['onehot', 'ordinal', 'mixed']
        ), "Invalid categorical encoding method"
        assert (
            self.numerical_encoding in ['standard', 'robust', 'quantile', 'yeo-johnson', 'none']
        ), "Invalid numerical scaling method"
        assert self.target_feature in self.data.columns, "Target feature not found in data"
        
        cols_union = set(self.numerical_features + self.categorical_features + [self.target_feature])
        data_cols = set(self.data.columns)
        
        if len(cols_union - data_cols) > 0:
            raise ValueError(f"Some features in config are missing in the data: {cols_union - data_cols}")
        
        if len(data_cols - cols_union) > 0:
            raise ValueError(f"Some features in data are missing in config: {data_cols - cols_union}")

        assert isinstance(self.n_jobs, int), "Invalid number of jobs"
    
    def _post_processing(self):
        
        ##################################################################################################   
        # Deduplicate Features
        for col in self.data.columns:
            if col != self.target_feature:
                if self.data[col].nunique() == 1:
                    self.data.drop(columns=[col], inplace=True)
                    # remove columns from numerical and categorical features
                    if col in self.numerical_features:
                        self.numerical_features.remove(col)
                    elif col in self.categorical_features:
                        self.categorical_features.remove(col)
        
        ##################################################################################################   
        # Target Encoding
        target = self.data[self.target_feature].copy()
        if self.task_type == 'classification':
            self.num_classes = target.nunique()
            self.clf_type = 'binary' if self.num_classes == 2 else 'multiclass'
            target_array, codes = pd.factorize(target)
            target = pd.Series(target_array, index=target.index, name=self.target_feature)
            self.target_codes = {i: code for i, code in enumerate(codes)}
        elif self.task_type == 'regression':
            self.num_classes = None
            self.clf_type = None
            self.target_codes = None
        elif self.task_type == 'survival':
            self.num_classes = None
            self.clf_type = None
            self.target_codes = None
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
        
        ##################################################################################################   
        # Numerical Encoding
        data_num = self.data[self.numerical_features].copy()
        
        if len(data_num) > 0:
            if self.numerical_encoding == 'none':
                pass
            else:
                if self.numerical_encoding == 'standard':
                    numerical_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('minmax', MinMaxScaler())
                    ])
                elif self.numerical_encoding == 'robust':
                    numerical_pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('minmax', MinMaxScaler())
                    ])
                elif self.numerical_encoding == 'quantile':
                    if len(data_num) > 1000:
                        numerical_pipeline = Pipeline([
                            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=1000)),
                            ('minmax', MinMaxScaler())
                        ])
                    else:
                        numerical_pipeline = Pipeline([
                            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=len(data_num))),
                            ('minmax', MinMaxScaler())
                        ])
                elif self.numerical_encoding == 'yeo-johnson':
                    numerical_pipeline = Pipeline([
                        ('scaler', PowerTransformer(method='yeo-johnson')),
                        ('minmax', MinMaxScaler())
                    ])
                else:
                    raise ValueError(f"Invalid numerical encoding method: {self.numerical_encoding}")

                numerical_pipeline.set_output(transform='pandas')
                data_num = numerical_pipeline.fit_transform(data_num)
                self.numerical_features = data_num.columns.tolist()

        ##################################################################################################   
        # Categorical Encoding
        data_cat = self.data[self.categorical_features].copy().astype(str)
        
        if len(data_cat) > 0:
                    
            if self.categorical_encoding == 'onehot':
                categorical_encoder = OneHotEncoder(
                    drop = 'first', sparse_output=False, max_categories=20, handle_unknown='error',
                )
                self.one_hot_cols = data_cat.columns.tolist()
            
            elif self.categorical_encoding == 'ordinal':
                categorical_encoder = OrdinalEncoder()
            
            elif self.categorical_encoding == 'mixed':
                # High cardinality categorical features are encoded using OrdinalEncoder
                # Low cardinality categorical features are encoded using OneHotEncoder
                threshold = 50
                n_unique_categories = data_cat.nunique().sort_values(ascending=False)
                high_cardinality_features = n_unique_categories[n_unique_categories > threshold].index.tolist()
                low_cardinality_features = n_unique_categories[n_unique_categories <= threshold].index.tolist()

                high_cardinality_encoder = OrdinalEncoder(
                    encoded_missing_value=np.nan, handle_unknown='error'
                )
                
                low_cardinality_encoder = OneHotEncoder(
                    drop = 'first', sparse_output=False, max_categories=20, handle_unknown='error', 
                )
                
                self.one_hot_cols = low_cardinality_features
                
                transformers = [
                    ('cat_high', high_cardinality_encoder, high_cardinality_features),
                    ('cat_low', low_cardinality_encoder, low_cardinality_features)
                ]
                
                categorical_encoder = ColumnTransformer(transformers=transformers, n_jobs=self.n_jobs)
            else:
                raise ValueError(f"Invalid categorical encoding method: {self.categorical_encoding}")
            
            categorical_encoder.set_output(transform='pandas')
            data_cat = categorical_encoder.fit_transform(data_cat)
            data_cat.columns = categorical_encoder.get_feature_names_out()
            
            self.categorical_features = data_cat.columns.tolist()
            self.categorical_feature_codes = {
                feature: list(data_cat[feature].unique()) for feature in data_cat.columns
            }
        
        ##################################################################################################   
        # Final Data
        del self.data
        self.data = pd.concat([data_num, data_cat, target], axis=1)
        
        return self.data
    
    def show_dataset_info(self):
        missing_ratio = self.data.isna().sum().sum() / (self.data.shape[0] * self.data.shape[1])
        print("="*100)
        print(f"Task name: {self.task_name}  Task type: {self.task_type}")
        print(f"Target: {self.target_feature} Num classes: {self.num_classes}")
        print(f'Data Shape: {self.data.shape} (num {len(self.numerical_features)} cat {len(self.categorical_features)})')
        print(f'Missing ratio: {missing_ratio*100:4.1f}%')
        print(f"Feature Groups:")
        if len(self.feature_groups) > 0:
            for feature_group, features in self.feature_groups.items():
                if len(features) > 6:
                    print(f"    - {feature_group}: {len(features)} features (e.g., {','.join(features[:3])} ... {','.join(features[-3:])})")
                else:
                    print(f"    - {feature_group}: {len(features)} features ({','.join(features)})")
        else:
            print("None")
        print("="*100)
        
    def _update_feature_groups(self):
        
        if len(self.feature_groups) == 0:
            return
        
        updated_feature_groups = {group_name: [] for group_name in self.feature_groups.keys()}
        
        ##################################################################################################
        # for each feature, find the group name
        for col in self.data.columns:
            
            # for target feature, skip
            if col == self.target_feature:
                continue
            
            # categorize features
            if col in self.categorical_features:
                
                # ordinal features, name is the original feature name
                if col not in self.one_hot_cols:
                    col_name_original = col
                # one-hot features, name is the original feature name + '_' + category
                else:
                    col_name_original = col.split('_')[:-1]
                    col_name_original = '_'.join(col_name_original)
                
                for group_name, features in self.feature_groups.items():
                    if col_name_original in features:
                        updated_feature_groups[group_name].append(col)
            
            # numerical features, name is the original feature name
            elif col in self.numerical_features:
                for group_name, features in self.feature_groups.items():
                    if col in features:
                        updated_feature_groups[group_name].append(col)
            else:
                raise ValueError(f"Invalid feature: {col}")
        
        ##################################################################################################
        # deduplicate feature groups
        for group_name, features in updated_feature_groups.items():
            updated_feature_groups[group_name] = list(set(features))
        
        ##################################################################################################   
        # Validation
        # Check for intersections between feature group sets
        feature_group_sets = [set(item) for item in updated_feature_groups.values()]
        for i in range(len(feature_group_sets)):
            for j in range(i + 1, len(feature_group_sets)):
                intersection = feature_group_sets[i] & feature_group_sets[j]
                if len(intersection) > 0:
                    raise ValueError(f"Found overlapping features between feature groups: {intersection}")
        
        # Check union of feature groups matches data columns (excluding target)
        data_features = set(self.data.columns.tolist()) - {self.target_feature}
        feature_group_features = set().union(*feature_group_sets) if feature_group_sets else set()
        
        missing_features = data_features - feature_group_features
        extra_features = feature_group_features - data_features
        
        if len(missing_features) > 0:
            raise ValueError(f"Features in data but missing from feature groups: {missing_features}")
        if len(extra_features) > 0:
            raise ValueError(f"Features in feature groups but not in data: {extra_features}")
        
        self.feature_groups = updated_feature_groups
        
        

