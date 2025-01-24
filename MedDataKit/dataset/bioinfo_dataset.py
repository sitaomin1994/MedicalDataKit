import pandas as pd
import os
import pyreadr
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from config import DATA_DIR, DATA_DOWNLOAD_DIR
from scipy.io import arff

from ..downloader import OpenMLDownloader
from ..downloader import RDataDownloader
from ..downloader import KaggleDownloader
from ..downloader import UCIMLDownloader
from .base_dataset import Dataset
from .base_raw_dataset import RawDataset
from .base_ml_task_dataset import MLTaskDataset, MLTaskPreparationConfig
from ..data_pipe_routines.missing_data_routines import BasicMissingDataHandler
from ..data_pipe_routines.data_type_routines import BasicFeatureTypeHandler
from ..utils import handle_targets


def count_show_unique(data):
    for col in data.columns:
        unique_values = data[col].unique().tolist()
        data_type = data[col].dtype
        num_unique = data[col].nunique()
        if num_unique < 20:
            print(f"{col} ({data_type}) => {num_unique} values: {unique_values}")
        else:
            print(f"{col} ({data_type}) => {num_unique} values")


###################################################################################################
# Codon Dataset
###################################################################################################
class CodonUsageDataset(Dataset):

    def __init__(self):
        
        name = 'codon'
        subject_area = 'Medical'
        year = 2020
        url = 'https://archive.ics.uci.edu/dataset/577/codon+usage'
        download_link = 'https://archive.ics.uci.edu/static/public/577/codon+usage.zip'
        description = "DNA codon usage frequencies of a large sample of diverse biological organisms from different taxa"
        notes = 'Bioinformatics'
        data_type = 'numerical'
        source = 'uci'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            data_type = data_type,
            source = source
        )
        
        self.data_dir = os.path.join(DATA_DIR, self.name)
        self.raw_dataset: RawDataset = None
        self.ml_ready_dataset: MLReadyDataset = None
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset and specify meta data information
        
        Returns:
            raw_data: pd.DataFrame, raw data
            meta_data: dict, meta data
        """
        # download data
        downloader = UCIMLDownloader(url = self.download_link)
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data        
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'codon_usage.csv'), sep=',', low_memory=False)
        raw_data = raw_data.drop(['SpeciesID', 'Ncodons', 'SpeciesName', 'DNAtype'], axis=1)
        
        raw_data.replace("non-B hepatitis virus", np.nan, inplace=True)
        raw_data.replace("12;I", np.nan, inplace=True)
        raw_data.replace('-', np.nan, inplace=True)

        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = ['Kingdom']
        binary_features = []
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Kingdom']
        sensitive_features = []
        drop_features = []
        task_names = ['predict_kingdom']
        
        feature_groups = {}
        fed_cols = []
        
        return {
            'numerical_features': numerical_features,
            'binary_features': binary_features,
            'multiclass_features': multiclass_features,
            'ordinal_features': ordinal_features,
            'target_features': target_features,
            'sensitive_features': sensitive_features,
            'drop_features': drop_features,
            'feature_groups': feature_groups,
            'task_names': task_names,
            'ordinal_feature_order_dict': ordinal_feature_order_dict,
            'fed_cols': fed_cols
        }
    
    def _set_target_feature(
        self, data: pd.DataFrame, raw_data_config: dict, task_name: str, drop_unused_targets: bool
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Set target feature based on task name
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            drop_unused_targets: bool, whether to drop unused target features
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """

        if task_name == 'predict_kingdom':
            target_info = {
                'target': 'Kingdom',
                'task_type': 'classification'
            }
        
        assert target_info['target'] in data.columns, f"target {target_info['target']} is not in data columns"
        
        return data, target_info
    
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Set target feature based on task name
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """
        return data, {
            'numerical_features': data_config['numerical_features'],
            'categorical_features': [],
        }
        
    
    def _handle_missing_data(self, data: pd.DataFrame, categorical_features: list) -> Tuple[pd.DataFrame, dict]:
        """
        Handle missing data
        
        Args:
            data: pd.DataFrame, raw data
            categorical_features: list, categorical features
            
        Returns:
            data: pd.DataFrame, processed data
            missing_data_info: dict, missing data processing information
        """
        return data.dropna(), {}

 
###################################################################################################################################
# GENE3494 Dataset
###################################################################################################################################
class GENE3494Dataset(Dataset):

    def __init__(self, k = 30):
        
        name = 'gene3494'
        subject_area = 'Medical'
        year = 2020
        url = 'GEO'
        download_link = ''
        description = "GEO Dataset 3494"
        notes = 'Bioinformatics'
        data_type = 'numerical'
        source = 'geo'
        self.k = k  # k - number of gene features to use from each platform
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            data_type = data_type,
            source = source
        )
        
        self.data_dir = os.path.join(DATA_DIR, self.name)
        self.raw_dataset: RawDataset = None
        self.ml_ready_dataset: MLReadyDataset = None
        
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset and specify meta data information
        
        Returns:
            raw_data: pd.DataFrame, raw data
            meta_data: dict, meta data
        """
        # load local files from data_dir
        df_gene96 = pd.read_csv(os.path.join(self.data_dir, 'genotype_data_gpl96.csv'))
        df_gene97 = pd.read_csv(os.path.join(self.data_dir, 'genotype_data_gpl97.csv'))
        ttest96 = pd.read_csv(os.path.join(self.data_dir, 'ttest_results_gpl96.csv'))
        ttest97 = pd.read_csv(os.path.join(self.data_dir, 'ttest_results_gpl97.csv'))
        df_pheno = pd.read_csv(os.path.join(self.data_dir, 'phenotype_data.csv'))
        
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
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        binary_features = [
            'p53_seq_mut_status', 'p53.DLDA_classifier_result', 'PgR_status', 'DSS_EVENT'
        ]
        multiclass_features = [
            'DLDA_error', 'Elston_histologic_grade', 'ER_status', 'Lymph_node_status'
        ]
        
        numerical_features = [
            col for col in raw_data.columns if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['DSS_EVENT']
        sensitive_features = ['age_at_diagnosis']
        drop_features = []
        task_names = ['predict_dss_event']
        
        feature_groups = {}
        fed_cols = []
        
        return {
            'numerical_features': numerical_features,
            'binary_features': binary_features,
            'multiclass_features': multiclass_features,
            'ordinal_features': ordinal_features,
            'target_features': target_features,
            'sensitive_features': sensitive_features,
            'drop_features': drop_features,
            'feature_groups': feature_groups,
            'task_names': task_names,
            'ordinal_feature_order_dict': ordinal_feature_order_dict,
            'fed_cols': fed_cols
        }
        
    def _set_target_feature(
        self, data: pd.DataFrame, raw_data_config: dict, task_name: str, drop_unused_targets: bool
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Set target feature based on task name
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            drop_unused_targets: bool, whether to drop unused target features
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """

        if task_name == 'predict_dss_event':
            target_info = {
                'target': 'DSS_EVENT',
                'task_type': 'classification'
            }
        
        assert target_info['target'] in data.columns, f"target {target_info['target']} is not in data columns"
        
        return data, target_info
    
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Set target feature based on task name
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """
        return data, {
            'numerical_features': data_config['numerical_features'],
            'categorical_features': [],
        }
        
    
    def _handle_missing_data(self, data: pd.DataFrame, categorical_features: list) -> Tuple[pd.DataFrame, dict]:
        """
        Handle missing data
        
        Args:
            data: pd.DataFrame, raw data
            categorical_features: list, categorical features
            
        Returns:
            data: pd.DataFrame, processed data
            missing_data_info: dict, missing data processing information
        """
        return data.dropna(), {}