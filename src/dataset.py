import pandas as pd
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self):
        
        # raw data and processed data
        self.raw_data: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None

        # meta data for raw data
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

        # meta data for processed data
        self.num_rows_ = 0
        self.num_cols_ = 0
        self.sensitive_features_ = []
        self.numerical_features_ = []
        self.categorical_features_ = []
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
        # check all columns are included in the meta data
        assert len(self.numerical_features) + len(self.categorical_features) + len(self.ordinal_features) == len(data.columns) - 1

        # reset index
        data.reset_index(drop=True, inplace=True)

        # convert numerical features to float type
        data[self.numerical_features] = data[self.numerical_features].astype(float)

        # convert ordinal features to int type
        data[self.ordinal_features] = data[self.ordinal_features].astype(int)

        # convert categorical features to int type
        data[self.categorical_features] = data[self.categorical_features].astype(int)

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
        print(f"Categorical features: {self.categorical_features}")
        print(f"Target feature: {self.target_feature}")
        print(f"Target type: {self.target_type}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Target codes mapping: {self.target_codes_mapping}")

    
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
            print(f"Feature '{feature}': {count} missing values")
        print("\nMissing pattern statistics:")
        print(f'Total missing patterns: {len(self.missing_pattern_stats)}')
        print('Top 10 missing patterns:')
        for pattern, count in list(self.missing_pattern_stats.items())[:10]:
            print(f"Pattern '{pattern}': {count*100:.2f} %")

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


from .downloader import OpenMLDownloader
from .downloader import RDataDownloader

class DermatologyDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        downloader = OpenMLDownloader(data_id=35)
        raw_data = downloader.download()

        # specify meta data
        self.sensitive_features = ['Age']
        self.numerical_features = ['Age']
        self.ordinal_features = [col for col in raw_data.columns[:-1] if col not in self.numerical_features]
        self.categorical_features = []
        self.target_feature = raw_data.columns[-1]
        self.target_type = 'multiclass'
        
        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()
    

class SupportDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        downloader = RDataDownloader(dataset_path='casebase/data/support.rda', package_url = 'https://cran.r-project.org/src/contrib/casebase_0.10.6.tar.gz')
        raw_data = downloader.download()

        # specify meta data
        self.sensitive_features = ['age', 'race', 'sex']
        self.categorical_features = ['sex', 'dzgroup', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
        self.target_feature = 'death'
        self.target_type = 'binary'
        self.numerical_features = [col for col in raw_data.columns if col not in self.categorical_features + [self.target_feature]]
        self.ordinal_features = []

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    
# todo: download and save data and avoid loading it every time -> check if the data is already downloaded
# todo: add codes for categorical features
# todo: kaggle data downloader - kaggle api key

class KidneyDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        pass

    def handle_missing_data(self, data: pd.DataFrame):
        pass
    
    def custom_download(self):
        import requests
        import zipfile
        import io
        import tempfile
        import os
        import mimetypes
        import patoolib
        import arff
        import pandas as pd
        url = 'https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip'
        response = requests.get(url)
        print(response.headers.get('Content-Type'))
        print(mimetypes.guess_type(url))
        zip_content = io.BytesIO(response.content)
        print(zip_content)
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # uncompress rar file in temp_dir
        rar_file_name = 'Chronic_Kidney_Disease.rar'
        rar_file_path = os.path.join(temp_dir, rar_file_name)
        patoolib.extract_archive(rar_file_path, outdir=temp_dir, interactive=False)
        # remove rar file
        os.remove(rar_file_path)

        # read arff file
        arff_file_path = os.path.join(temp_dir, 'Chronic_Kidney_Disease', 'chronic_kidney_disease_full.arff')
        data = pd.read_csv(arff_file_path)

        return data
    


class BreastCancerLjubljanaDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        pass

    def handle_missing_data(self, data: pd.DataFrame):
        pass

    def custom_download(self):
        import requests
        import zipfile
        import tempfile
        import io
        import os
        import shutil
        url = 'https://archive.ics.uci.edu/static/public/14/breast+cancer.zip'
        response = requests.get(url)
        zip_content = io.BytesIO(response.content)
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        file_name = 'breast-cancer.data'
        file_path = os.path.join(temp_dir, file_name)
        if file_name.endswith('.data'):
            data = pd.read_csv(file_path, header=None)

        # remove temp_dir
        shutil.rmtree(temp_dir)

        return data
        
class BreastCancerWisconsinDataset(Dataset):

    def __init__(self):
        super().__init__()

    def load(self):
        pass

    def handle_missing_data(self, data: pd.DataFrame):
        pass

    def custom_download(self):
        import requests
        import zipfile
        import tempfile
        import io
        import os
        import shutil
        url = 'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip'
        response = requests.get(url)
        zip_content = io.BytesIO(response.content)  
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print(os.listdir(temp_dir))
        
        file_name = 'wdbc.data'
        file_path = os.path.join(temp_dir, file_name)
        if file_name.endswith('.data'):
            data = pd.read_csv(file_path, header=None)
        
        # remove temp_dir
        shutil.rmtree(temp_dir)

        return data
            
    
    
        




        

