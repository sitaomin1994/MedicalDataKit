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
    URLDownloader
)
from .base_dataset import Dataset
from .base_raw_dataset import RawDataset
from .base_ml_task_dataset import MLTaskDataset, MLTaskPreparationConfig
from ..data_pipe_routines.missing_data_routines import BasicMissingDataHandler
from ..data_pipe_routines.data_type_routines import BasicFeatureTypeHandler
from ..utils import handle_targets
    

###################################################################################################################################
# Bacteremia Dataset
###################################################################################################################################
class BacteremiaDataset(Dataset):

    def __init__(self):
        
        name = 'bacteremia'
        subject_area = 'Medical'
        year = 2020
        url = 'https://hbiostat.org/data/'
        download_link = 'https://zenodo.org/api/records/7554815/files-archive'
        description = "Bacteremia dataset"
        notes = 'Bacteremia'
        data_type = 'numerical'
        self.pub_link = 'A Risk Prediction Model for Screening Bacteremic Patients: A Cross Sectional Study; ' \
                        'Regression with Highly Correlated Predictors: Variable Omission Is Not the Solution'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            data_type = data_type
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
        if not os.path.exists(os.path.join(self.data_dir, 'Bacteremia_public_S2.csv')):
            downloader = URLDownloader(url = self.download_link, zipfile = True)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'Bacteremia_public_S2.csv'), index_col=0).reset_index(drop=True)
        
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
            "SEX", "BloodCulture"
        ]
        multiclass_features = []
        
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['BloodCulture']
        sensitive_features = ["SEX", "AGE"]
        drop_features = []
        task_names = ['predict_BloodCulture']
        
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

        if task_name == 'predict_BloodCulture':
            target_info = {
                'target': 'BloodCulture',
                'task_type': 'classification'
            }
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
