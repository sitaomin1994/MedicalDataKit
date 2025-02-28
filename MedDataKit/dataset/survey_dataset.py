import pandas as pd
import os
import pyreadr
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from config import DATA_DIR, DATA_DOWNLOAD_DIR
from scipy.io import arff
import rdata

from ..downloader import (
    OpenMLDownloader,
    RDataDownloader,
    KaggleDownloader,
    UCIMLDownloader,
    URLDownloader,
    LocalDownloader
)
from .base_dataset import Dataset
from .base_raw_dataset import RawDataset
from .base_ml_task_dataset import MLTaskDataset, MLTaskPreparationConfig
from ..data_pipe_routines.missing_data_routines import BasicMissingDataHandler
from ..data_pipe_routines.data_type_routines import BasicFeatureTypeHandler
from ..utils import handle_targets

###################################################################################################################################
# CDC Diabetics Dataset
###################################################################################################################################
class CDCDiabetesDataset(Dataset):

    def __init__(self, version: str = '1'):
        
        name = 'cdc_diabetes'
        subject_area = 'Medical'
        year = 2022
        url = 'https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset'
        url2 = 'https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators'
        download_link = None
        description = "CDC Diabetes Dataset"
        notes = 'CAD'
        data_type = 'mixed'
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
        self.version = version
        if version == '1':
            self.file_name = 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
        elif version == '2':
            self.file_name = 'diabetes_binary_health_indicators_BRFSS2015.csv'
        else:
            raise ValueError(f'Invalid version: {version}')
        
    def _load_raw_data(self):
        
        # download data
        downloader = KaggleDownloader(
            dataset_name = 'alexteboul/diabetes-health-indicators-dataset',
            file_names = [self.file_name],
            download_all = True
        )
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, self.file_name))
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
    
        # Specify meta data
        numerical_features = [
            'BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income'
        ]
        multiclass_features = []
        ordinal_features = []
        ordinal_feature_order_dict = {}
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Diabetes_binary']
        sensitive_features = ['Age', 'Sex']
        drop_features = []
        task_names = ['predict_diabetes']
        
        feature_groups = {}
        fed_cols = []
        
        return {
            'numerical_features': numerical_features,
            'binary_features': binary_features,
            'multiclass_features': multiclass_features,
            'ordinal_features': ordinal_features,
            'ordinal_feature_order_dict': ordinal_feature_order_dict,
            'target_features': target_features,
            'sensitive_features': sensitive_features,
            'drop_features': drop_features,
            'feature_groups': feature_groups,
            'task_names': task_names,
            'fed_cols': fed_cols
        }

    def _set_target_feature(
        self, data: pd.DataFrame, raw_data_config: dict, task_name: str, drop_unused_targets: bool
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Set target feature based on task name 
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration
            task_name: str, task name
            drop_unused_targets: bool, whether to drop unused target features
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """
        if task_name == 'predict_diabetes':
            
            target_info = {
                'target': 'Diabetes_binary',
                'task_type': 'classification'
            }
        else:
            raise ValueError(f"Invalid task name: {task_name}")
        
        # TODO: reformat this in the future
        if drop_unused_targets == True:
            print(f"For this dataset, drop_unused_targets True is considered as False. No target features will be dropped.")
        
        assert (target_info['target'] in data.columns), "Target feature not found in data"
        
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
        ordinal_as_numerical = ml_task_prep_config.ordinal_as_numerical
        feature_type_handler = BasicFeatureTypeHandler(ordinal_as_numerical)
        
        data, numerical_features, categorical_features = feature_type_handler.handle_feature_type(
            data, data_config
        )
        
        return data, {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
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
# NHANES GH
###################################################################################################################################
class NHANESGHDataset(Dataset):

    def __init__(self):
        
        name = 'rhc'
        subject_area = 'Medical'
        year = 2024
        url = 'https://hbiostat.org/data/repo/nhgh'
        download_link = 'https://hbiostat.org/data/repo/nhgh.tsv'
        description = "Extracted from NHANES"
        notes = 'Survey'
        data_type = 'mixed'
        self.pub_link = None
        source = 'vdb'
        
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
        if not os.path.exists(os.path.join(self.data_dir, 'nhgh.csv')):
            downloader = URLDownloader(url = self.download_link, zipfile = False)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'nhgh.tsv'), index_col=0, delimiter = '\t').reset_index(drop = True)
        
        # Map income ranges to their mean values
        income_map = {
            '>= 100000': 100000,
            '[25000,35000)': 30000,
            '[35000,45000)': 40000, 
            '[75000,100000)': 87500,
            '[20000,25000)': 22500,
            '[10000,15000)': 12500,
            '[45000,55000)': 50000,
            '[15000,20000)': 17500,
            '[55000,65000)': 60000,
            '[5000,10000)': 7500,
            '[65000,75000)': 70000,
            '[0,5000)': 2500,
            '> 20000': 25000,
            '< 20000': 15000
        }
        raw_data['income'] = raw_data['income'].map(income_map)
        

        
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
            'sex', 'tx', 'dx'
        ]
        multiclass_features = ['re']
        
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['tx', 'dx']
        sensitive_features = ['age', 'sex', 're']
        drop_features = []
        task_names = ['predict_tx', 'predict_dx']
        
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

        if task_name == 'predict_tx':
            target_info = {
                'target': 'tx',
                'task_type': 'classification'
            }
            data = data.drop(columns=['dx'])
        elif task_name == 'predict_dx':
            target_info = {
                'target': 'dx',
                'task_type': 'classification'
            }
            data = data.drop(columns=['tx'])
        else:
            raise ValueError(f"task name {task_name} is not supported")
        
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
        
        ordinal_as_numerical = ml_task_prep_config.ordinal_as_numerical
        feature_type_handler = BasicFeatureTypeHandler(ordinal_as_numerical)
        
        data, numerical_features, categorical_features = feature_type_handler.handle_feature_type(
            data, data_config
        )
        
        return data, {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
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
        missing_data_handler = BasicMissingDataHandler(
            threshold1_num = 0.4,
            threshold2_num = 0.05,
            threshold1_cat = 0.4,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info