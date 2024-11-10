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


class MyocardialInfarctionDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'myocardial_infarction'
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data()
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data()

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['AGE', 'SEX']
        self.drop_features = []
        self.ordinal_features = [
            'STENOK_AN', 'DLIT_AG',  'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n',
            'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n'
        ]
        self.binary_features = [
            'SEX',
            'IBS_NASL', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08',
            'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10',
            'endocr01', 'endocr02', 'endocr03', 
            'zableg01', 'zableg02', 'zableg03', 'zableg04', 'zableg06',
            'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST',
            'IM_PG_P', 
            'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08',
            'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06',
            'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 
            'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07',
            'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12',
            'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08',
            'GIPO_K', 'GIPER_Na',
            'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S',
            'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n',
            # target
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV',
            'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN'
            ]
        self.multiclass_features = [
            'FK_STENOK', 'IBS_POST', 'GB', 'SIM_GIPERT', 'ZSN_A',
            'ant_im', 'lat_im', 'inf_im', 'post_im', 
            # target
            'LET_IS'
        ]
        
        self.target_features = [
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV',
            'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN',  'LET_IS'
        ]
        self.numerical_features = [
           'AGE', 'INF_ANAM', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 
           'K_BLOOD',  'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self):
        downloader = UCIMLDownloader(
            url = 'https://archive.ics.uci.edu/static/public/579/myocardial+infarction+complications.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'MI.data'), header=None, index_col = 0, na_values = ['?'])
        raw_data.reset_index(inplace = True, drop = True)
        columns = [
            'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK',
            'IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 
            'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08',
            'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10',
            'endocr01', 'endocr02', 'endocr03', 
            'zableg01', 'zableg02', 'zableg03', 'zableg04', 'zableg06',
            'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT',
            'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 
            'ant_im', 'lat_im', 'inf_im', 'post_im', 
            'IM_PG_P', 
            'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08',
            'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06',
            'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 
            'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07',
            'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12',
            'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08',
            'GIPO_K',
            'K_BLOOD', 
            'GIPER_Na',
            'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 
            'TIME_B_S',
            'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 
            'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 
            'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 
            'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n',
            'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n',
            # target
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV',
            'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 
            'LET_IS'
        ]
        
        raw_data.columns = columns
        
        self.feature_groups = {
            'patient_info': [
                'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 
                'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 
                'endocr01', 'endocr02', 'endocr03', 'zableg01', 'zableg02', 'zableg03', 'zableg04', 'zableg06'
            ],
            'clinical_info': [
                'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 
                'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P', 'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 
                'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08', 'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 
                'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 
                'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 
                'n_p_ecg_p_11', 'n_p_ecg_p_12'
            ],
            'input_info': [
                'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 
                'GIPO_K', 'K_BLOOD', 'GIPER_Na', 'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S', 
                'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 
                'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 
                'TRENT_S_n'
            ]
        }

        return raw_data
    
    def get_ml_ready_data(self, task_name: str = None):
        
        from MedDataKit.data_pipe_routines.data_type_routines import basic_data_type_formulation, update_data_type_info
        from MedDataKit.data_pipe_routines.data_encoding_routines import basic_categorical_encoding
        from MedDataKit.data_pipe_routines.data_encoding_routines import basic_numerical_encoding
        from MedDataKit.data_pipe_routines.missing_data_routines import basic_missing_mitigation
        
        data = self.raw_data.copy()
        data_preprocessing_log = {
            'raw_data_shape': data.shape
        }
                
        if task_name is None:
            task_name = 'lethal_outcome'
        
        if task_name == 'lethal_outcome':
            
            # specify target and task type
            target = 'LET_IS'
            task_type = 'classification'
            num_classes = data[target].nunique()
            
            # specify drop features
            drop_features = self.target_features.copy()
            drop_features.remove('LET_IS')
            
        elif task_name == 'lethal_outcome_binary':
            
            # specify target and task type
            data['LET_IS_binary'] = data[target].apply(lambda x: 0 if x == 0 else 1)
            target = 'LET_IS_binary'
            task_type = 'classification'
            num_classes = 2
            
            # specify drop features
            drop_features = self.target_features.copy()
            
        elif task_name == 'chronic_heart_failure':
            
            # specify target and task type
            target = 'ZSN'
            task_type = 'classification'
            num_classes = 2
            
            # specify drop features
            drop_features = self.target_features.copy()
            drop_features.remove('ZSN')
            
        elif task_name == 'num_complications':
            
            # specify target and task type
            complications = self.target_features.copy()
            complications.remove('LET_IS')
            data['num_complications'] = data.apply(lambda row: sum(row[complications]), axis = 1)
            target = 'num_complications'
            task_type = 'regression'
            num_classes = None
            
            # specify drop features
            drop_features = self.target_features.copy()
        else:
            raise ValueError(f"Task {task_name} is not supported")
        
        ########################################################################################
        # drop features
        ########################################################################################
        data.drop(columns = drop_features, inplace = True)

        ########################################################################################
        # data type formulation
        ########################################################################################
        data, data_type_info = basic_data_type_formulation(
            data = data,
            numerical_cols = self.numerical_features,
            ordinal_cols = self.ordinal_features,
            binary_cols = self.binary_features,
            multiclass_cols = self.multiclass_features,
            target_col = target
        )
                
        ########################################################################################
        # missing data mitigation
        ########################################################################################
        data, missing_data_info = basic_missing_mitigation(data, threshold1 = 0.15)
        
        data_type_info = update_data_type_info(data, data_type_info)
        numerical_features = data_type_info['numerical_feature']
        categorical_features = data_type_info['categorical_feature']
        
        ########################################################################################
        # data encoding
        ########################################################################################
        data, numerical_encoding_info = basic_numerical_encoding(data, numerical_features)
        # print(data_encoding_info)
        # data, data_encoding_info = basic_categorical_encoding(data, categorical_features)
        # print(data_encoding_info)
        
        data_type_info = update_data_type_info(data, data_type_info)
        numerical_features = data_type_info['numerical_feature']
        categorical_features = data_type_info['categorical_feature']
        
        ########################################################################################
        # data configuration
        ########################################################################################
        data = data[numerical_features + categorical_features + [target]]
        data_config = {
            'numerical_feature': numerical_features,
            'categorical_feature': categorical_features,
            'target': target,
            'task_type': task_type,
            'num_classes': num_classes,
        }
        
        data_preprocessing_log.update({
            'data_type_info': data_type_info,
            'missing_data_info': missing_data_info,
            'numerical_encoding_info': numerical_encoding_info,
            'processed_data_shape': data.shape
        })
        
        ##########################################################################################
        # Logging
        ##########################################################################################
        print('Data preprocessing log:')
        print("Raw data shape: ", data_preprocessing_log['raw_data_shape'])  # todo: add more details
        print("Processed data shape: ", data_preprocessing_log['processed_data_shape'])
        
        return data, data_config, data_preprocessing_log
    
    
class GENE3494Dataset(Dataset):

    def __init__(self, data_dir, k = 30):
        super().__init__()
        self.data_dir = data_dir
        self.name = 'GENE3494'
        self.k = k  # k - number of gene features to use from each platform
        self.data_dir = os.path.join(DATA_DIR, self.name)

    def load_raw_data(self):
        raw_data = self.download_data(self.data_dir)
        return raw_data

    def load(self):

        # download data
        raw_data = self.download_data(self.data_dir)

        # specify meta data
        self.raw_data = raw_data
        self.sensitive_features = ['age_at_diagnosis']
        self.drop_features = []
        self.ordinal_features = []
        self.binary_features = [
            'p53_seq_mut_status', 'p53.DLDA_classifier_result', 'PgR_status', 'DSS_EVENT'
        ]
        self.multiclass_features = [
            'DLDA_error', 'Elston_histologic_grade', 'ER_status', 'Lymph_node_status'
        ]
        
        self.target_features = [
            'DSS_EVENT'
        ]
        self.numerical_features = [
            col for col in raw_data.columns if col not in self.binary_features + self.multiclass_features + self.ordinal_features
        ]

        # basic processing
        self.raw_data = self.basic_processing(raw_data)

    def handle_missing_data(self, data: pd.DataFrame):
        return data.dropna()

    def download_data(self, data_dir: str):
        
        # load local files from data_dir
        df_gene96 = pd.read_csv(os.path.join(data_dir, 'genotype_data_gpl96.csv'))
        df_gene97 = pd.read_csv(os.path.join(data_dir, 'genotype_data_gpl97.csv'))
        ttest96 = pd.read_csv(os.path.join(data_dir, 'ttest_results_gpl96.csv'))
        ttest97 = pd.read_csv(os.path.join(data_dir, 'ttest_results_gpl97.csv'))
        df_pheno = pd.read_csv(os.path.join(data_dir, 'phenotype_data.csv'))
        
        ttest96.sort_values(by = 'p_value', inplace = True, ascending = True)
        feature_set96 = ttest96.loc[:self.k, 'feature'].tolist()
        
        ttest97.sort_values(by = 'p_value', inplace = True, ascending = True)
        feature_set97 = ttest97.loc[:self.k, 'feature'].tolist()
        
        df_gene96 = df_gene96[['ID'] + feature_set96]
        df_gene97 = df_gene97[['ID'] + feature_set97]
        df = pd.merge(df_gene96, df_gene97, on = 'ID')
        df = pd.merge(df, df_pheno, on = 'ID')
        df.drop(columns = ['ID'], inplace = True)
        
        phenotype_features = df_pheno.columns.tolist()
        phenotype_features.remove('ID')
        phenotype_features.remove('DSS_EVENT')
        self.feature_groups = {
            'phenotype': phenotype_features,
            'gene_96': feature_set96,
            'gene_97': feature_set97
        }
        
        raw_data = df

        return raw_data
    
    def get_ml_ready_data(self, task_name: str = None):
        
        from MedDataKit.data_pipe_routines.data_type_routines import basic_data_type_formulation, update_data_type_info
        from MedDataKit.data_pipe_routines.data_encoding_routines import basic_categorical_encoding
        from MedDataKit.data_pipe_routines.data_encoding_routines import basic_numerical_encoding
        from MedDataKit.data_pipe_routines.missing_data_routines import basic_missing_mitigation
        
        data = self.raw_data.copy()
        data_preprocessing_log = {
            'raw_data_shape': data.shape
        }
                
        if task_name is None:
            task_name = 'lethal_outcome'
        
        if task_name == 'lethal_outcome':
            
            # specify target and task type
            target = 'LET_IS'
            task_type = 'classification'
            num_classes = data[target].nunique()
            
            # specify drop features
            drop_features = self.target_features.copy()
            drop_features.remove('LET_IS')
            
        elif task_name == 'lethal_outcome_binary':
            
            # specify target and task type
            data['LET_IS_binary'] = data[target].apply(lambda x: 0 if x == 0 else 1)
            target = 'LET_IS_binary'
            task_type = 'classification'
            num_classes = 2
            
            # specify drop features
            drop_features = self.target_features.copy()
            
        elif task_name == 'chronic_heart_failure':
            
            # specify target and task type
            target = 'ZSN'
            task_type = 'classification'
            num_classes = 2
            
            # specify drop features
            drop_features = self.target_features.copy()
            drop_features.remove('ZSN')
            
        elif task_name == 'num_complications':
            
            # specify target and task type
            complications = self.target_features.copy()
            complications.remove('LET_IS')
            data['num_complications'] = data.apply(lambda row: sum(row[complications]), axis = 1)
            target = 'num_complications'
            task_type = 'regression'
            num_classes = None
            
            # specify drop features
            drop_features = self.target_features.copy()
        else:
            raise ValueError(f"Task {task_name} is not supported")
        
        ########################################################################################
        # drop features
        ########################################################################################
        data.drop(columns = drop_features, inplace = True)

        ########################################################################################
        # data type formulation
        ########################################################################################
        data, data_type_info = basic_data_type_formulation(
            data = data,
            numerical_cols = self.numerical_features,
            ordinal_cols = self.ordinal_features,
            binary_cols = self.binary_features,
            multiclass_cols = self.multiclass_features,
            target_col = target
        )
                
        ########################################################################################
        # missing data mitigation
        ########################################################################################
        data, missing_data_info = basic_missing_mitigation(data, threshold1 = 0.15)
        
        data_type_info = update_data_type_info(data, data_type_info)
        numerical_features = data_type_info['numerical_feature']
        categorical_features = data_type_info['categorical_feature']
        
        ########################################################################################
        # data encoding
        ########################################################################################
        data, numerical_encoding_info = basic_numerical_encoding(data, numerical_features)
        # print(data_encoding_info)
        # data, data_encoding_info = basic_categorical_encoding(data, categorical_features)
        # print(data_encoding_info)
        
        data_type_info = update_data_type_info(data, data_type_info)
        numerical_features = data_type_info['numerical_feature']
        categorical_features = data_type_info['categorical_feature']
        
        ########################################################################################
        # data configuration
        ########################################################################################
        data = data[numerical_features + categorical_features + [target]]
        data_config = {
            'numerical_feature': numerical_features,
            'categorical_feature': categorical_features,
            'target': target,
            'task_type': task_type,
            'num_classes': num_classes,
        }
        
        data_preprocessing_log.update({
            'data_type_info': data_type_info,
            'missing_data_info': missing_data_info,
            'numerical_encoding_info': numerical_encoding_info,
            'processed_data_shape': data.shape
        })
        
        ##########################################################################################
        # Logging
        ##########################################################################################
        print('Data preprocessing log:')
        print("Raw data shape: ", data_preprocessing_log['raw_data_shape'])  # todo: add more details
        print("Processed data shape: ", data_preprocessing_log['processed_data_shape'])
        
        return data, data_config, data_preprocessing_log
    
