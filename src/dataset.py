import pandas as pd
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self):
        
        # raw data and processed data
        self.raw_data: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None

        # meta data
        self.num_rows = 0
        self.num_cols = 0
        self.sensitive_features = []
        self.numerical_features = []
        self.ordinal_features = []
        self.categorical_features = []
        self.target_feature = None
        self.target_type = None
        self.num_classes = None
        self.target_codes_mapping = None

        # missing data information
        self.num_missing_values = 0
        self.missing_value_stats = {}  # missing features with its missing values
        self.missing_pattern_stats = {}  # missing patterns with its ratio
    
    @abstractmethod
    def load(self):
        """
        Load dataset as raw data and fetch meta data information such as num_cols, num_rows, num_classes, 
        numerical_features, categorical_features, target_feature
        """
        pass


from .downloader import OpenMLDownloader
class DermatologyDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        downloader = OpenMLDownloader(data_id=35)
        self.raw_data = downloader.download()
    
        # fetch meta data and basic processing
        self.num_rows, self.num_cols = self.raw_data.shape
        self.sensitive_features = ['Age']

        # numerical features 
        self.numerical_features = ['Age']
        self.raw_data[self.numerical_features] = self.raw_data[self.numerical_features].astype(float)

        # ordinal features
        self.ordinal_features = [col for col in self.raw_data.columns[:-1] if col not in self.numerical_features]
        self.raw_data[self.ordinal_features] = self.raw_data[self.ordinal_features].astype(int)

        # categorical features
        self.categorical_features = []
        self.raw_data[self.categorical_features] = self.raw_data[self.categorical_features].astype(int)

        # target
        self.target_feature = self.raw_data.columns[-1]
        self.target_type = 'multiclass'
        self.num_classes = self.raw_data[self.target_feature].nunique()
        # convert target variable dtype
        if self.target_type in ['multiclass', 'binary']:
            self.raw_data[self.target_feature], self.target_codes_mapping = pd.factorize(self.raw_data[self.target_feature], sort=True)
            # store mapping as dict
            self.target_codes_mapping = dict(enumerate(self.target_codes_mapping))
        else:
            # if target variable is numerical, directly convert to float
            self.raw_data[self.target_feature] = self.raw_data[self.target_feature].astype(float)

    def show_meta_data(self):
        print(f"Number of rows: {self.num_rows}")
        print(f"Number of columns: {self.num_cols}")
        print(f"Sensitive features: {self.sensitive_features}")
        print(f"Numerical features: {self.numerical_features}")
        print(f"Ordinal features: {self.ordinal_features}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Target feature: {self.target_feature}")
        print(f"Target type: {self.target_type}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Target codes mapping: {self.target_codes_mapping}")

    
    def get_missing_data_statistics(self):
        
        # get missing data statistics
        self.num_missing_values = self.raw_data.isnull().sum().sum()
        self.missing_feature_table = self.raw_data.isnull().sum()
        self.missing_value_stats = self.missing_feature_table[self.missing_feature_table > 0].to_dict()
        # missing pattern statistics
        mask = self.raw_data.isnull()
        pattern_counts = mask.apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1).value_counts(normalize=True, sort=True)
        self.missing_pattern_stats = pattern_counts.to_dict()

    def show_missing_data_statistics(self):
        print(f"Number of missing values: {self.num_missing_values}")
        print("Missing value statistics:")
        for feature, count in self.missing_value_stats.items():
            print(f"Feature '{feature}': {count} missing values")
        print("\nMissing pattern statistics:")
        print(f'Total missing patterns: {len(self.missing_pattern_stats)}')
        print('Top 10 missing patterns:')
        for pattern, count in list(self.missing_pattern_stats.items())[:10]:
            print(f"Pattern '{pattern}': {count*100:.2f} %")
    


