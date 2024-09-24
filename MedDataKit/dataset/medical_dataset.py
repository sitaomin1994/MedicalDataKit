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


def count_show_unique(data):
    for col in data.columns:
        unique_values = data[col].unique().tolist()
        data_type = data[col].dtype
        num_unique = data[col].nunique()
        if num_unique < 20:
            print(f"{col} ({data_type}) => {num_unique} values: {unique_values}")
        else:
            print(f"{col} ({data_type}) => {num_unique} values")

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
        self.multiclass_features = [raw_data.columns[-1]]
        self.target_features = [raw_data.columns[-1]]
        
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
        self.binary_features = ['sex', 'diabetes', 'dementia', 'death']
        self.multiclass_features = ['dzgroup', 'dzclass', 'race', 'ca']
        self.ordinal_features = []
        self.target_features = ['death']
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features
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
        self.binary_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        self.multiclass_features = []
        self.target_features = ['classification']
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features
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
    


# class BreastCancerLjubljanaDataset(Dataset):

#     def __init__(self):
#         super().__init__()

#     def load(self):
#         pass

#     def handle_missing_data(self, data: pd.DataFrame):
#         pass

#     def custom_download(self):
#         import requests
#         import zipfile
#         import tempfile
#         import io
#         import os
#         import shutil
#         url = 'https://archive.ics.uci.edu/static/public/14/breast+cancer.zip'
#         response = requests.get(url)
#         zip_content = io.BytesIO(response.content)
#         temp_dir = tempfile.mkdtemp()
#         with zipfile.ZipFile(zip_content) as zip_ref:
#             zip_ref.extractall(temp_dir)
        
#         file_name = 'breast-cancer.data'
#         file_path = os.path.join(temp_dir, file_name)
#         if file_name.endswith('.data'):
#             data = pd.read_csv(file_path, header=None)

#         # remove temp_dir
#         shutil.rmtree(temp_dir)

#         return data
        
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
        self.binary_features = ['1']
        self.multiclass_features = []
        self.target_features = ['1']
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features
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
        self.binary_features = ['status']
        self.multiclass_features = []
        self.target_features = ['status']
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features
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
        self.sensitive_features = ['age', 'sex']
        self.drop_features = []
        self.ordinal_features = []
        self.binary_features = ['sex']
        self.multiclass_features = []
        self.target_features = ['total_UPDRS', 'motor_UPDRS']
        self.numerical_features = [
            col for col in raw_data.columns 
            if col not in self.binary_features + self.multiclass_features + self.ordinal_features + self.drop_features
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


class AutismChildDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'autism_child'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        count_show_unique(raw_data)
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['age', 'gender', 'ethnicity']
        self.drop_features = ['age_desc']
        self.ordinal_features = []
        self.multiclass_features = ['ethnicity', 'contry_of_res', 'relation']
        self.numerical_features = ['age', 'result']
        self.binary_features = [
            col for col in raw_data.columns 
            if col not in self.numerical_features + self.multiclass_features + self.ordinal_features + self.drop_features
        ]
        self.target_features = ['Class/ASD']

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/419/autistic+spectrum+disorder+screening+data+for+children.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        from scipy.io import arff

        raw_data, meta = arff.loadarff(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'Autism-Child-Data.arff'))
        
        raw_data = pd.DataFrame(raw_data)
        raw_data = raw_data.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        raw_data.replace('?', np.nan, inplace = True)

        return raw_data
    

class AutismAdultDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'autism_adult'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        count_show_unique(raw_data)
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['age', 'gender', 'ethnicity']
        self.drop_features = ['age_desc']
        self.ordinal_features = []
        self.multiclass_features = ['ethnicity', 'contry_of_res', 'relation']
        self.numerical_features = ['age', 'result']
        self.binary_features = [
            col for col in raw_data.columns 
            if col not in self.numerical_features + self.multiclass_features + self.ordinal_features + self.drop_features
        ]
        self.target_features = ['Class/ASD']

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        from scipy.io import arff

        raw_data, meta = arff.loadarff(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'Autism-Adult-Data.arff'))
        
        raw_data = pd.DataFrame(raw_data)
        raw_data = raw_data.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        raw_data.replace('?', np.nan, inplace = True)

        return raw_data
    

class AutismAdolescentDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'autism_adolescent'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        count_show_unique(raw_data)
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['age', 'gender', 'ethnicity']
        self.drop_features = ['age_desc']
        self.ordinal_features = []
        self.multiclass_features = ['ethnicity', 'contry_of_res', 'relation']
        self.numerical_features = ['age', 'result']
        self.binary_features = [
            col for col in raw_data.columns 
            if col not in self.numerical_features + self.multiclass_features + self.ordinal_features + self.drop_features
        ]
        self.target_features = ['Class/ASD']

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/420/autistic+spectrum+disorder+screening+data+for+adolescent.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        from scipy.io import arff

        raw_data, meta = arff.loadarff(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'Autism-Adolescent-Data.arff'))
        
        raw_data = pd.DataFrame(raw_data)
        raw_data = raw_data.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        raw_data.replace('?', np.nan, inplace = True)

        return raw_data
    


class DiabeticKaggleDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'diabetic_kaggle'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        return raw_data

    def load(self):
        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['race', 'gender', 'age']
        self.drop_features = ['encounter_id']
        self.ordinal_features = []
        self.binary_features = [
            'gender', 'acetohexamide', 'glipizide-metformin', 'glimepiride-pioglitazone', 
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']
        self.numerical_features = [
            'age', 'weight', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
            'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]

        self.multiclass_features = [
            col for col in raw_data.columns if col not in self.binary_features + self.numerical_features + self.drop_features
        ]

        self.target_features = ['readmitted']

        # basic processing
        self.raw_data = self.custom_raw_data_preprocessing(raw_data)
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = KaggleDownloader(
            dataset_name='brandao/diabetes',
            file_names = ['diabetic_data.csv'],
            download_all = True,
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'diabetic_data.csv'), na_values = ['?',"Unknown/Invalid"], low_memory = False)

        return raw_data
    
    def custom_raw_data_preprocessing(self, raw_data: pd.DataFrame):

        "ML ready data reference: https://www.kaggle.com/code/chongchong33/predicting-hospital-readmission-of-diabetics/notebook#2.1-Transform-data-type"

        # mapping age and weight to numerical values
        raw_data['age'] = raw_data['age'].map({
            '[90-100)': 95,
            '[80-90)': 85,
            '[70-80)': 75,
            '[60-70)': 65,
            '[50-60)': 55,
            '[40-50)': 45,
            '[30-40)': 35,    
            '[20-30)': 25,
            '[10-20)': 15,
            '[0-10)': 5,
        })

        raw_data['weight'] = raw_data['weight'].map({
            '>200': 200,
            '[175-200)': 187.5,
            '[150-175)': 162.5,
            '[125-150)': 137.5,
            '[100-125)': 112.5,
            '[75-100)': 87.5,
            '[50-75)': 62.5,
            '[25-50)': 37.5,
            '[0-25)': 12.5,
        })

        # ordinal encoding 
        raw_data['A1Cresult'] = pd.Categorical(raw_data['A1Cresult'], categories = ['Norm', '>7', '>8'], ordered = True)
        raw_data['max_glu_serum'] = pd.Categorical(raw_data['max_glu_serum'], categories = ['Norm', '>200', '>300'], ordered = True)


        return raw_data
