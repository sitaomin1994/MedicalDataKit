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


###################################################################################################################################
# Codon Usage Dataset
###################################################################################################################################
class CodonUsageDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.name = 'codon_usage'
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
        self.multiclass_features = ['Kingdom']
        self.target_features = ['Kingdom']
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
            url = 'https://archive.ics.uci.edu/static/public/577/codon+usage.zip'
        )

        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, DATA_DOWNLOAD_DIR, 'codon_usage.csv'), sep=',', low_memory=False)
        raw_data = raw_data.drop(['SpeciesID', 'Ncodons', 'SpeciesName', 'DNAtype'], axis=1)
        
        raw_data.replace("non-B hepatitis virus", np.nan, inplace=True)
        raw_data.replace("12;I", np.nan, inplace=True)
        raw_data.replace('-', np.nan, inplace=True)

        return raw_data

 
###################################################################################################################################
# GENE3494 Dataset
###################################################################################################################################
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
            task_name = 'DSS_EVENT'
        
        if task_name == 'DSS_EVENT':
            
            # specify target and task type
            target = 'DSS_EVENT'
            task_type = 'classification'
            num_classes = data[target].nunique()
            
            # specify drop features
            drop_features = self.target_features.copy()
            drop_features.remove('DSS_EVENT')
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
        
        data_type_info = update_data_type_info(data, data_type_info, target)
        numerical_features = data_type_info['numerical_feature']
        categorical_features = data_type_info['categorical_feature']
        
        ########################################################################################
        # data encoding
        ########################################################################################
        data, numerical_encoding_info = basic_numerical_encoding(data, numerical_features)
        # print(data_encoding_info)
        # data, data_encoding_info = basic_categorical_encoding(data, categorical_features)
        # print(data_encoding_info)
        
        data_type_info = update_data_type_info(data, data_type_info, target)
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