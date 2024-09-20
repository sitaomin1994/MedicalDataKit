from ..downloader import OpenMLDownloader
from ..downloader import RDataDownloader
from ..downloader import KaggleDownloader
from ..downloader import UCIMLDownloader
from .base_dataset import Dataset
import pandas as pd
import os
import pyreadr
import numpy as np

from config import DATA_DIR, DATA_DOWNLOAD_DIR

class DermatologyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'dermatology'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load(self):
        
        # download data
        raw_data = self.download_data()

        # specify meta data
        self.sensitive_features = ['Age']
        self.drop_features = []
        self.numerical_features = ['Age']
        self.ordinal_features = [col for col in raw_data.columns[:-1] if col not in self.numerical_features + self.drop_features]
        self.binary_features = []
        self.multiclass_features = []
        self.target_feature = raw_data.columns[-1]
        self.target_type = 'multiclass'
        
        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()
    
    def download_data(self):
        # download data
        downloader = OpenMLDownloader(data_id=35)
        print(self.data_dir)
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load data
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'data.csv')) # todo: data.csv as parameters
        return raw_data


class SupportDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'support'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.sensitive_features = ['age', 'race', 'sex']
        self.drop_features = []
        self.binary_features = ['sex', 'diabetes', 'dementia']
        self.multiclass_features = ['dzgroup', 'dzclass', 'race', 'ca']
        self.ordinal_features = []
        self.target_feature = 'death'
        self.target_type = 'binary'
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features + [self.target_feature]
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        # download data
        downloader = RDataDownloader(
            dataset_path_in_package='casebase/data/support.rda', package_url = 'https://cran.r-project.org/src/contrib/casebase_0.10.6.tar.gz'
        )
        
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        data_file_name = 'support.rda'
        data_file_path = os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, data_file_name)
        # use pyreadr to read the .rda file
        result = pyreadr.read_r(data_file_path)
        
        # pyreadr returns a dictionary, we take the value of the first key as the DataFrame
        raw_data = next(iter(result.values()))

        # replace empty strings with NA
        for col in raw_data.columns:
            if raw_data[col].dtype.name == 'category':
                raw_data[col] = raw_data[col].astype(object).replace('', pd.NA).astype('category')
            else:
                raw_data[col] = raw_data[col].replace('', pd.NA)
        
        return raw_data

    
# todo: add codes for categorical features

class KidneyDiseaseDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'kidney_disease'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load(self):
        
        # download data
        raw_data = self.download_data()

        # specify meta data
        self.sensitive_features = ['age']
        self.drop_features = ['id']
        self.ordinal_features = ['al', 'su']
        self.binary_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        self.multiclass_features = []
        self.target_feature = 'classification'
        self.target_type = 'binary'
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features + [self.target_feature]
        ]

        self.raw_data = raw_data
        self.raw_data = self.custom_raw_data_preprocessing(self.raw_data)
        # basic processing
        self.raw_data = self.basic_processing(self.raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()
    
    def download_data(self):
        downloader = KaggleDownloader(
            dataset_name='mansoordaku/ckdisease',
            file_names = ['kidney_disease.csv'],
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'kidney_disease.csv'))

        return raw_data
    
    def custom_raw_data_preprocessing(self, raw_data: pd.DataFrame):

        # categories error correction
        corretion_cols = ['dm', 'cad']
        for col in corretion_cols:
            raw_data[col] = raw_data[col].replace({' yes':  'yes', '\tyes': 'yes', '\tno': 'no'})
        
        raw_data['classification'] = raw_data['classification'].replace({'ckd\t': 'ckd'})

        # numerical feature error correction
        corretion_cols = ['pcv', 'wc', 'rc']
        for col in corretion_cols:
            raw_data[col] = raw_data[col].replace('\t?', np.nan)
            raw_data[col] = raw_data[col].replace('\t[0-9]+', lambda x: x.group(0).replace('\t', ''))
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

        return raw_data
    


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
        self.name = 'breast_cancer_wisconsin'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = []
        self.drop_features = []
        self.ordinal_features = []
        self.binary_features = []
        self.multiclass_features = []
        self.target_feature = '1'
        self.target_type = 'binary'
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features + [self.target_feature]
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'wdbc.data'), header=None, index_col = 0)
        raw_data.columns = [str(i) for i in raw_data.columns]

        return raw_data
    
class ParkinsonsUCIDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'parkinsons_uci'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = []
        self.drop_features = []
        self.ordinal_features = []
        self.binary_features = []
        self.multiclass_features = []
        self.target_feature = 'status'
        self.target_type = 'binary'
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features + [self.target_feature]
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/174/parkinsons.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'parkinsons.data'), index_col = 0)
        raw_data.columns = [str(i) for i in raw_data.columns]

        return raw_data
    
class ParkinsonsTelemonitoringDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'parkinsons_telemonitoring'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = []
        self.drop_features = []
        self.ordinal_features = []
        self.binary_features = []
        self.multiclass_features = []
        self.target_feature = 'status'
        self.target_type = 'binary'
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features + [self.target_feature]
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'parkinsons_updrs.data'), index_col = 0)
        raw_data.columns = [str(i) for i in raw_data.columns]

        return raw_data
