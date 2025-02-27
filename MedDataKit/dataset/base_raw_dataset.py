import pandas as pd
import numpy as np
from MedDataKit.utils import column_check
from typing import Dict, Any, Optional, Tuple

class RawDataset:
    
    def __init__(
        self, 
        data: pd.DataFrame,
        # Feature types
        numerical_features: list,
        binary_features: list,
        multiclass_features: list,
        ordinal_features: list,
        ordinal_feature_order_dict: dict,
        # Meta data
        target_features: list,
        sensitive_features: list,
        drop_features: list,
        # ML Tasks
        task_names: list,
        # Federated Information
        fed_cols: list = None,
        feature_groups: dict = None,
    ):
        
        # Feature types
        self.numerical_features = numerical_features
        self.ordinal_features = ordinal_features
        self.ordinal_feature_order_dict = ordinal_feature_order_dict
        self.binary_features = binary_features
        self.multiclass_features = multiclass_features
        
        # Meta data for raw data
        self.drop_features = drop_features
        self.sensitive_features = sensitive_features
        self.target_features = target_features
        
        # Federated Information
        if fed_cols is not None:
            self.fed_cols = fed_cols
        else:
            self.fed_cols = []
        
        if feature_groups is not None:
            self.feature_groups = feature_groups
        else:
            self.feature_groups = {}
        
        # ML Tasks
        self.task_names = task_names
        
        # Basic Processing the raw data
        self.data = data.reset_index(drop=True)
        self._meta_data_validation(self.data)
        self._basic_processing(self.data)
        
        # Statistics
        self.num_rows = 0
        self.num_cols = 0
        self.feature_stats = {}
        self.column_info_ = {}
        
        # Missing Data Statistics
        self.mask: pd.DataFrame = None
        self.missing_ratio = None
        self.missing_feature_stats = {}
        self.missing_pattern_stats = {}
        
        # Calculate Data Statistics
        self.num_rows, self.num_cols = self.data.shape
        self.calculate_feature_stats(self.data)
        self.calculate_missing_data_stats(self.data)
    
    def get_data(self):
        if self.data is None:
            raise ValueError("Data is not loaded")
        else:
            return self.data
    
    @property
    def raw_data_config(self):
        assert (len(self.task_names) > 0), "No task names specified for raw data"
        assert (len(self.target_features) > 0), "No target features specified for raw data"
        
        return {
            'numerical_features': self.numerical_features,
            'binary_features': self.binary_features,
            'multiclass_features': self.multiclass_features,
            'ordinal_features': self.ordinal_features,
            'ordinal_feature_order_dict': self.ordinal_feature_order_dict,
            'target_features': self.target_features,
            'drop_features': self.drop_features,
            'sensitive_features': self.sensitive_features,
            'feature_groups': self.feature_groups,
            'task_names': self.task_names,
            'fed_cols': self.fed_cols
        }
    
    @property
    def categorical_features_stats(self):
        if len(self.feature_stats) == 0:
            raise ValueError("Feature statistics are not calculated")
        return {
            feature: self.feature_stats[feature] for feature 
            in self.ordinal_features + self.binary_features + self.multiclass_features
        }
    
    @property
    def numerical_features_stats(self):
        if len(self.feature_stats) == 0:
            raise ValueError("Feature statistics are not calculated")
        return {feature: self.feature_stats[feature] for feature in self.numerical_features}
    
    @property
    def target_feature_stats(self):
        if len(self.feature_stats) == 0:
            raise ValueError("Feature statistics are not calculated")
        return {feature: self.feature_stats[feature] for feature in self.target_features}
    
    @property
    def missing_data_stats(self):
        return {
            'missing_ratio': self.missing_ratio,
            'missing_feature_stats': self.missing_feature_stats,
            'missing_pattern_stats': self.missing_pattern_stats
        }
        
    @property
    def data_stats(self):
        return {
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'feature_stats': self.feature_stats
        }
    
    @property
    def column_info(self):
        if self.data is None:
            raise ValueError("Data is not loaded")
        else:
            if len(self.column_info_) == 0:
                self.column_info_ = column_check(self.data)
            else:
                return self.column_info_
    
    def _meta_data_validation(self, data: pd.DataFrame):
        """
        Validate correctness of specified meta data
        """
        # Verify column types are not overlapped and cover all features of the dataset
        all_features = set(
            self.numerical_features + self.ordinal_features + self.binary_features + self.multiclass_features
        )
        assert (
            len(all_features) == (
                len(self.numerical_features) + len(self.ordinal_features) + 
                len(self.binary_features) + len(self.multiclass_features)
            )
        ), "Features overlap between different feature types"
        
        # Check if all features are in the data
        missing_cols1 = all_features - set(data.columns)
        missing_cols2 = set(data.columns) - all_features
        
        assert (
            len(missing_cols1) == 0 and len(missing_cols2) == 0
        ), f"Some columns are not included in meta data: {missing_cols1} and {missing_cols2}"
        
        # Verify all elements of the target features, sensitive features, and drop features are in the data
        not_in_features = [feature for feature in self.sensitive_features if feature not in data.columns]
        assert len(not_in_features) == 0, f"Some specified sensitive features are not in the data: {not_in_features}"
        
        not_in_features = [feature for feature in self.target_features if feature not in data.columns]
        assert len(not_in_features) == 0, f"Some specified target features are not in the data: {not_in_features}"
        
        not_in_features = [feature for feature in self.drop_features if feature not in data.columns]
        assert len(not_in_features) == 0, f"Some specified drop features are not in the data: {not_in_features}"
        
        assert (
            len(self.target_features) > 0
        ), "No target features specified"
        
        ###########################################################################################################
        # Check correctness of feature groups and fed_cols
        ###########################################################################################################
        feature_groups = self.feature_groups
        fed_cols = self.fed_cols
        target_features = self.target_features
        
        # Check if fed_cols are in the data
        if len(fed_cols) > 0:
            assert all(item in data.columns for item in fed_cols), "Some specified fed_cols are not in the data"
        
        if len(feature_groups) > 0:
            # Check for intersections between feature group sets
            feature_group_sets = [set(item) for item in feature_groups.values()]
            for i in range(len(feature_group_sets)):
                for j in range(i + 1, len(feature_group_sets)):
                    intersection = feature_group_sets[i] & feature_group_sets[j]
                    if len(intersection) > 0:
                        raise ValueError(f"Found overlapping features between feature groups: {intersection}")
            
            # Check union of feature groups matches raw data columns
            raw_features = set(data.columns.tolist())
            feature_group_features = set().union(*feature_group_sets)
            missing_features = raw_features - feature_group_features
            extra_features = feature_group_features - raw_features
            
            if len(missing_features) > 0:
                raise ValueError(f"Features in raw data but missing from feature groups: {missing_features}")
            if len(extra_features) > 0:
                raise ValueError(f"Features in feature groups but not in raw data: {extra_features}")
        
    def _basic_processing(self, data: pd.DataFrame):
        """
        Conduct basic processing for raw data based on the meta data
        """
        # Convert numerical features to numeric type
        data[self.numerical_features] = data[self.numerical_features].astype(float)

        # Convert ordinal features to string type
        for feature in self.ordinal_features:
            if feature in self.ordinal_feature_order_dict:
                order = self.ordinal_feature_order_dict[feature]
                data[feature] = data[feature].astype(pd.CategoricalDtype(order, ordered=True))
            else:
                data[feature] = data[feature].astype(str)
        
        # Convert binary features to string type
        data[self.binary_features] = data[self.binary_features].astype(str)
        data[self.binary_features] = data[self.binary_features].replace('nan', np.nan)

        # Convert multiclass features to string type
        data[self.multiclass_features] = data[self.multiclass_features].astype(str)
        data[self.multiclass_features] = data[self.multiclass_features].replace('nan', np.nan)
        
        # for feature in self.multiclass_features:
        #     data[feature], codes = pd.factorize(data[feature], sort=True)
        #     data[feature] = data[feature].replace(-1, np.nan)
        #     self.feature_codes[feature] = dict(enumerate(codes))

        # Move target variables to the end of the dataframe
        for target_feature in self.target_features:
            data = data.drop(columns=[target_feature]).assign(**{target_feature: data[target_feature]})
        
        self.data = data
        
        return data
    
    @staticmethod
    def factorize_ordinal_features(data: pd.DataFrame, ordinal_features: list = None):
        
        data_copy = data.copy()
        ordinal_feature_codes = {}

        for feature in ordinal_features:
            data_copy[feature], codes = pd.factorize(data_copy[feature], sort=True)
            data_copy[feature] = data_copy[feature].replace(-1, np.nan)
            ordinal_feature_codes[feature] = dict(enumerate(codes))
        
        return data_copy, ordinal_feature_codes
    
    def calculate_feature_stats(self, data: pd.DataFrame):
        """
        Calculate feature statistics
        """
        for feature in self.numerical_features:
            self.feature_stats[feature] = {
                'min': data[feature].min(),
                'max': data[feature].max(),
                'mean': data[feature].mean(),
                'std': data[feature].std(),
                'na_ratio': data[feature].isna().sum() / len(data)
            }
        
        for feature in self.ordinal_features:
            value_counts = data[feature].value_counts().to_dict()
            self.feature_stats[feature] = {
                'unique_values': list(value_counts.keys()),
                'value_counts': value_counts,
                'order': list(value_counts.keys()),
                'na_ratio': data[feature].isna().sum() / len(data)
            }
        
        for feature in self.binary_features + self.multiclass_features:
            self.feature_stats[feature] = {
                'unique_values': list(data[feature].unique()),
                'value_counts': data[feature].value_counts().to_dict(),
                'na_ratio': data[feature].isna().sum() / len(data)
            }
    
    def calculate_missing_data_stats(self, data: pd.DataFrame):
        """
        Calculate missing data statistics
        """
        # get missing data statistics
        self.mask = pd.isna(data)
        self.missing_ratio = self.mask.sum().sum()/(data.shape[0]*data.shape[1])
        
        # missing feature statistics
        missing_feature_table = self.mask.sum()/data.shape[0]
        self.missing_feature_stats = missing_feature_table[missing_feature_table > 0].to_dict()
        
        # missing pattern statistics
        mask = data.isnull()
        pattern_counts = mask.apply(
            lambda x: ''.join(x.astype(int).astype(str)), axis=1
        ).value_counts(normalize=True, sort=True)
        self.missing_pattern_stats = pattern_counts.to_dict()              
    
    def show_dataset_info(
        self, 
        log_dir: str = None
    ):
        """
        Show all information of raw dataset
        """
        if log_dir is None:
            # Dimension
            print(f"Number of rows: {self.num_rows}, Number of columns: {self.num_cols}")
            
            # Features
            if len(self.numerical_features) > 20:
                print(f"Numerical features: ({len(self.numerical_features)}): [ {', '.join(self.numerical_features[:20])}"
                      f" ...... {', '.join(self.numerical_features[-20:])} ]")
            else:
                print(f"Numerical features: ({len(self.numerical_features)}): {self.numerical_features}")
                
            if len(self.ordinal_features) > 20:
                print(f"Ordinal features: ({len(self.ordinal_features)}): [ {', '.join(self.ordinal_features[:20])}"
                      f" ...... {', '.join(self.ordinal_features[-20:])} ]")
            else:
                print(f"Ordinal features: {self.ordinal_features}")
                
            if len(self.binary_features) > 20:
                print(f"Binary features: ({len(self.binary_features)}): [ {','.join(self.binary_features[:20])}"
                      f" ...... {','.join(self.binary_features[-20:])} ]")
            else:
                print(f"Binary features: {self.binary_features}")
                
            if len(self.multiclass_features) > 20:
                print(f"Multiclass features: ({len(self.multiclass_features)}): [ {','.join(self.multiclass_features[:20])}"
                      f" ...... {','.join(self.multiclass_features[-20:])} ]")
            else:
                print(f"Multiclass features: {self.multiclass_features}")
                
            print(f"Target features:")
            for target_feature in self.target_features:
                if target_feature in self.numerical_features:
                    target_type = 'numerical'
                elif target_feature in self.ordinal_features:
                    target_type = 'ordinal'
                elif target_feature in self.binary_features:
                    target_type = 'binary'
                elif target_feature in self.multiclass_features:
                    target_type = 'multiclass'
                else:
                    raise ValueError(f"Target feature {target_feature} is not numerical, binary, or multiclass")
                
                if target_type != 'numerical':
                    categories = self.target_feature_stats[target_feature]['unique_values']
                    if len(categories) > 20:
                        print(f"    - {target_feature} ({target_type}) => {len(categories)} categories")
                    else:
                        print(f"    - {target_feature} ({target_type}) => {categories}")
                else:
                    num_values = self.data[target_feature].nunique()
                    print(f"    - {target_feature} ({target_type}) => {num_values} values")
            
            print(f"Sensitive features: {self.sensitive_features}")
            print(f"Drop features: {self.drop_features}")
            
            # Feature Statistics
            print(f"Feature Distribution:")
            for feature in self.numerical_features:
                num_values = self.data[feature].nunique()
                min_value = self.numerical_features_stats[feature]['min']
                max_value = self.numerical_features_stats[feature]['max']
                mean_value = self.numerical_features_stats[feature]['mean']
                std_value = self.numerical_features_stats[feature]['std']
                na_ratio = self.numerical_features_stats[feature]['na_ratio']
                data_type = self.data[feature].dtype.name
                
                if len(feature) > 10:
                    feature = feature[:7] + '...'
                else:
                    feature = feature
                
                print(f"    - {feature[:10]:10s} ({data_type:5s}) : NA: {na_ratio*100:4.1f}% - {num_values:6d} values - "
                      f"[{min_value:8.2f}, {max_value:8.2f}] ({mean_value:8.2f}, {std_value:8.2f})")
            
            print(f"Feature categories (ordinal, binary, multiclass):")
            for feature_type, features in [
                ('ordinal', self.ordinal_features), 
                ('binary', self.binary_features), 
                ('multiclass', self.multiclass_features)
            ]:
                for feature in features:
                    if feature not in self.target_features:
                        
                        categories = self.feature_stats[feature]['unique_values']
                        na_ratio = self.feature_stats[feature]['na_ratio']
                        data_type = self.data[feature].dtype.name
                        
                        if len(feature) > 10:
                            feature = feature[:7] + '...'
                        else:
                            feature = feature
                        
                        if len(categories) > 20:
                            print(f"    - {feature:10s} ({feature_type}, {data_type:5s}): NA: {na_ratio*100:4.1f}% - {len(categories):4d} categories")
                        else:
                            print(f"    - {feature:10s} ({feature_type}, {data_type:5s}): NA: {na_ratio*100:4.1f}% - {categories}")
            
            # Feature Groups
            print(f"Feature Groups:")
            if len(self.feature_groups) > 0:
                for feature_group, features in self.feature_groups.items():
                    if len(features) > 6:
                        print(f"    - {feature_group}: {len(features)} features (e.g., {','.join(features[:3])} ... {','.join(features[-3:])})")
                    else:
                        print(f"    - {feature_group}: {len(features)} features ({','.join(features)})")
            else:
                print("None")
        else:
            raise NotImplementedError("Saving dataset information to a file is not implemented")
    
    def show_missing_data_statistics(self):
        """
        Show missing data statistics
        """
        print(f"Ratio of missing values: {self.missing_ratio}")
        print("Feature missing statistics:")
        missing_feature_stats = sorted(self.missing_feature_stats.items(), key=lambda x: x[1], reverse=True)
        for feature, ratio in missing_feature_stats:
            print(f"    {feature:20s}: {int(ratio*self.num_rows):4d} ({ratio*100:4.1f}%) missing values")
        print("\nMissing pattern statistics:")
        print(f'    Total missing patterns: {len(self.missing_pattern_stats)}')
        print('    Top 10 missing patterns:')
        for pattern, count in list(self.missing_pattern_stats.items())[:10]:
            print(f"    Pattern '{pattern}': {count*100:.2f} %")
    
    def visualize_missing_mask(self):
        raise NotImplementedError("Plotting missing mask is not implemented")
    
    def visualize_feature_distribution(self):
        raise NotImplementedError("Plotting feature distribution is not implemented")