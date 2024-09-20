import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self):
        
        # raw data and processed data
        self.raw_data: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None

        # meta data for raw data
        self.num_rows = 0
        self.num_cols = 0
        self.drop_features = []
        self.sensitive_features = []
        # we define four type of features
        self.numerical_features = []
        self.ordinal_features = []
        self.binary_features = []
        self.multiclass_features = []
        self.feature_codes = {}  # feature codes for categorical features - ordinal, binary, multiclass
        # target feature
        self.target_feature = None
        self.target_type = None
        self.num_classes = None
        self.target_codes_mapping = None

        # meta data for processed data
        self.num_rows_ = 0
        self.num_cols_ = 0
        self.sensitive_features_ = []
        # for processed data, we just have two data types - either numerical or categorical
        self.numerical_features_ = []
        self.categorical_features_ = []
        self.feature_codes_ = {}
        self.target_feature_ = None
        self.target_type_ = None
        self.num_classes_ = None
        self.target_codes_mapping_ = None

        # missing data information
        self.num_missing_values = 0
        self.missing_value_stats = {}  # missing features with its missing values
        self.missing_pattern_stats = {}  # missing patterns with its ratio
    
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

    def basic_processing(self, data: pd.DataFrame):
        """
        Conduct basic processing for raw data
        """
        data = data.reset_index(drop=True)
        # drop unnecessary features
        data.drop(columns=self.drop_features, inplace=True)
        
        # check all columns are included in the meta data
        print(len(self.numerical_features) + len(self.ordinal_features) + len(self.binary_features) + len(self.multiclass_features))
        print(len(data.columns) - 1)
        assert len(self.numerical_features) + len(self.ordinal_features) + len(self.binary_features) + len(self.multiclass_features) == len(data.columns) - 1

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

        # convert target variable dtype based on its type
        if self.target_type in ['multiclass', 'binary']:
            self.num_classes = data[self.target_feature].nunique()
            data[self.target_feature], self.target_codes_mapping = pd.factorize(data[self.target_feature], sort=True)
            # store mapping as dict
            self.target_codes_mapping = dict(enumerate(self.target_codes_mapping))
        else:
            self.num_classes = 0
            # if target variable is numerical, directly convert to float
            data[self.target_feature] = data[self.target_feature].astype(float)

        # move target variable to the end of the dataframe
        data = data.drop(columns=[self.target_feature]).assign(**{self.target_feature: data[self.target_feature]})
        
        # storenumber of rows and columns
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
        print(f"Target feature: {self.target_feature}")
        print(f"    Target type: {self.target_type}")
        print(f"    Number of classes: {self.num_classes}")
        print(f"    Target codes mapping: {self.target_codes_mapping}")
        print(f"Feature codes (ordinal, binary, multiclass):")
        for feature_type, features in [('ordinal', self.ordinal_features), ('binary', self.binary_features), ('multiclass', self.multiclass_features)]:
            for feature in features:
                print(f"    {feature} ({feature_type}): {self.feature_codes[feature]}")

    
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
            print(f"    {feature}: {count} missing values")
        print("\nMissing pattern statistics:")
        print(f'    Total missing patterns: {len(self.missing_pattern_stats)}')
        print('    Top 10 missing patterns:')
        for pattern, count in list(self.missing_pattern_stats.items())[:10]:
            print(f"    Pattern '{pattern}': {count*100:.2f} %")

    def preprocess(
            self,
            standardize: bool = False,
            normalize: bool = False,
            ordinal_as_numerical: bool = False,
            one_hot_categorical: bool = False,
            max_categories: int = 10
    ):
        """
        Preprocess the raw data to get ML ready data
        """
        
        self.processed_data = self.raw_data.copy()
        
        # missing data
        self.processed_data = self.handle_missing_data(self.processed_data)
        
        # standardization
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.processed_data[self.numerical_features] = scaler.fit_transform(self.processed_data[self.numerical_features])
        
        # normalization
        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            self.processed_data[self.numerical_features] = scaler.fit_transform(self.processed_data[self.numerical_features])
        
        # hanlding ordinal features - either consider them as numerical or categorical
        if ordinal_as_numerical:
            self.numerical_features.extend(self.ordinal_features)
            self.ordinal_features = []
        else:
            self.categorical_features.extend(self.ordinal_features)
            self.ordinal_features = []
        
        # one-hot encoding for categorical features
        if one_hot_categorical and len(self.categorical_features) > 0:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', max_categories=max_categories)
            encoded_features = encoder.fit_transform(self.processed_data[self.categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(self.categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=self.processed_data.index)
            self.processed_data = pd.concat([self.processed_data.drop(columns=self.categorical_features), encoded_df], axis=1)
        
            # update categorical features
            self.categorical_features = encoded_feature_names.tolist()
        
        # get meta data for processed data
        self.num_rows_, self.num_cols_ = self.processed_data.shape
        self.sensitive_features_ = self.sensitive_features
        self.numerical_features_ = self.numerical_features
        self.categorical_features_ = self.categorical_features
        self.target_feature_ = self.target_feature
        self.target_type_ = self.target_type
        self.num_classes_ = self.num_classes
        self.target_codes_mapping_ = self.target_codes_mapping
            
    
    
        




        

