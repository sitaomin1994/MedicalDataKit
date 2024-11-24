import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self):
        # todo: each feature should associated with a dictionary or class to store their properties
        
        ############################################################################################
        # raw data
        self.raw_data: pd.DataFrame = None

        # meta data for raw data
        self.num_rows = 0
        self.num_cols = 0
        self.drop_features = []
        self.sensitive_features = []
        self.target_features = []
        
        # we classify features into four types
        self.numerical_features = []
        self.ordinal_features = []
        self.binary_features = []
        self.multiclass_features = []
        self.feature_codes = {}  # feature codes for categorical features - ordinal, binary, multiclass

        ############################################################################################
        # ml-ready data
        self.ml_ready_data: pd.DataFrame = None
        
        # meta data for ml-ready data
        self.num_rows_mlready = 0
        self.num_cols_mlready = 0
        self.sensitive_features_mlready = []
        
        # for ml-ready data, we just have two data types - either numerical or categorical
        self.numerical_features_mlready = []
        self.categorical_features_mlready = []
        self.feature_codes_mlready = {}             # feature codes for categorical features
        self.target_feature_mlready = None          # target feature
        self.target_type_mlready = None             # target type
        self.num_classes_mlready = None             # number of classes
        self.target_codes_mapping_mlready = None    # target codes mapping

        # missing data information
        self.missing_data_cleaning_log = {}          # record the missing data cleaning steps for each feature
    
    @abstractmethod
    def load(self):
        """
        Load dataset as raw data and specify meta data information such as num_cols, num_rows, num_classes, 
        numerical_features, categorical_features, target_feature
        """
        pass

    @abstractmethod
    def handle_missing_data(self, data: pd.DataFrame):
        """
        Handle missing data in the dataset
        """
        pass

    def data_config(self):
        target_var = self.target_features[0]
        if target_var in self.numerical_features:
            task_type = 'regression'
        elif target_var in self.binary_features:
            task_type = 'binary'
        elif target_var in self.multiclass_features:
            task_type = 'multiclass'
        else:
            raise ValueError(f"Target feature {target_var} is not numerical, binary, or multiclass")
        
        numerical_features = [item for item in self.numerical_features if item != target_var]
        categorical_features = self.ordinal_features + self.binary_features + self.multiclass_features
        categorical_features = [item for item in categorical_features if item != target_var]

        return {
            'target_var': target_var,
            'sensitive_var': self.sensitive_features,
            'task_type': task_type,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
        }

    def basic_processing(self, data: pd.DataFrame):
        """
        Conduct basic processing for raw data
        """
        data = data.reset_index(drop=True)
        # drop unnecessary features
        data.drop(columns=self.drop_features, inplace=True)
        
        # check all columns are included in the meta data
        assert len(self.numerical_features) + len(self.ordinal_features) + len(self.binary_features) + len(self.multiclass_features) == len(data.columns)

        # reset index
        data.reset_index(drop=True, inplace=True)

        # convert numerical features to float type
        data[self.numerical_features] = data[self.numerical_features].astype(float)

        # convert ordinal features to int type
        for feature in self.ordinal_features:
            data[feature], codes = pd.factorize(data[feature], sort=True)
            data[feature] = data[feature].replace(-1, np.nan)
            self.feature_codes[feature] = dict(enumerate(codes))
        
        # convert binary features to int type
        for feature in self.binary_features:
            data[feature], codes = pd.factorize(data[feature], sort=True)
            data[feature] = data[feature].replace(-1, np.nan)
            self.feature_codes[feature] = dict(enumerate(codes))

        # convert multiclass features to int type
        for feature in self.multiclass_features:
            data[feature], codes = pd.factorize(data[feature], sort=True)
            data[feature] = data[feature].replace(-1, np.nan)
            self.feature_codes[feature] = dict(enumerate(codes))

        # move target variable to the end of the dataframe
        for target_feature in self.target_features:
            assert target_feature not in self.ordinal_features, "Target feature can only be numerical, binary, or multiclass"
            data = data.drop(columns=[target_feature]).assign(**{target_feature: data[target_feature]})
        
        # store number of rows and columns
        self.num_rows, self.num_cols = data.shape

        return data

    def show_meta_data(self):
        """
        Show meta data for raw data
        """
        print(f"Number of rows: {self.num_rows}")
        print(f"Number of columns: {self.num_cols}")
        print(f"Sensitive features: {self.sensitive_features}")
        print(f"Numerical features: {self.numerical_features}")
        print(f"Ordinal features: {self.ordinal_features}")
        print(f"Binary features: {self.binary_features}")
        print(f"Multiclass features: {self.multiclass_features}")
        print(f"Target features:")
        for target_feature in self.target_features:
            if target_feature in self.numerical_features:
                target_type = 'numerical'
            elif target_feature in self.binary_features:
                target_type = 'binary'
            elif target_feature in self.multiclass_features:
                target_type = 'multiclass'
            else:
                raise ValueError(f"Target feature {target_feature} is not numerical, binary, or multiclass")
            if target_type != 'numerical':
                print(f"    - {target_feature} ({target_type}) => {self.feature_codes[target_feature]}")
            else:
                print(f"    - {target_feature} ({target_type})")
        print(f"Feature codes (ordinal, binary, multiclass):")
        for feature_type, features in [
            ('ordinal', self.ordinal_features), 
            ('binary', self.binary_features), 
            ('multiclass', self.multiclass_features)
        ]:
            for feature in features:
                if feature not in self.target_features:
                    if len(self.feature_codes[feature]) > 20:
                        print(f"    - {feature} ({feature_type}): {len(self.feature_codes[feature])} categories")
                    else:
                        print(f"    - {feature} ({feature_type}): {self.feature_codes[feature]}")
        
        print(f"Feature Groups:")
        if hasattr(self, 'feature_groups'):
            for feature_group, features in self.feature_groups.items():
                if len(features) > 6:
                    print(f"    - {feature_group}: {len(features)} features (e.g., {','.join(features[:3])} ... {','.join(features[-3:])})")
                else:
                    print(f"    - {feature_group}: {len(features)} features ({','.join(features)})")

    
    def get_missing_data_statistics(self):
        """
        Get missing data statistics
        """
        # get missing data statistics
        self.num_missing_values = self.raw_data.isnull().sum().sum()
        self.missing_feature_table = self.raw_data.isnull().sum()
        self.missing_value_stats = self.missing_feature_table[self.missing_feature_table > 0].to_dict()
        
        # missing pattern statistics
        mask = self.raw_data.isnull()
        pattern_counts = mask.apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1).value_counts(normalize=True, sort=True)
        self.missing_pattern_stats = pattern_counts.to_dict()

    def show_missing_data_statistics(self):
        """
        Show missing data statistics
        """
        print(f"Number of missing values: {self.num_missing_values}")
        print("Missing value statistics:")
        for feature, count in self.missing_value_stats.items():
            print(f"    {feature}: {count} ({count/self.num_rows*100:.2f}%) missing values")
        print("\nMissing pattern statistics:")
        print(f'    Total missing patterns: {len(self.missing_pattern_stats)}')
        print('    Top 10 missing patterns:')
        for pattern, count in list(self.missing_pattern_stats.items())[:10]:
            print(f"    Pattern '{pattern}': {count*100:.2f} %")
    
    def get_ml_ready_data(self, task_name: str = None):
        """
        Preprocess the raw data to get ML ready data for a specific task
        """
        raise NotImplementedError
        
        # self.processed_data = self.raw_data.copy()
        
        # # missing data
        # self.processed_data = self.handle_missing_data(self.processed_data)
        
        # # hanlding ordinal features - either consider them as numerical or categorical
        # if ordinal_as_numerical:
        #     self.numerical_features.extend(self.ordinal_features)
        #     self.ordinal_features = []
        # else:
        #     self.categorical_features.extend(self.ordinal_features)
        #     self.ordinal_features = []
        
        # # get meta data for processed data
        # self.num_rows_, self.num_cols_ = self.processed_data.shape
        # self.sensitive_features_ = self.sensitive_features
        # self.numerical_features_ = self.numerical_features
        # self.categorical_features_ = self.categorical_features
        # self.target_feature_ = self.target_feature
        # self.target_type_ = self.target_type
        # self.num_classes_ = self.num_classes
        # self.target_codes_mapping_ = self.target_codes_mapping
            
    
    
        




        

