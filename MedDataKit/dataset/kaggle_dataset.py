import pandas as pd
import os
import pyreadr
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from config import DATA_DIR, DATA_DOWNLOAD_DIR
import warnings

from ..downloader import KaggleDownloader
from ..downloader import UCIMLDownloader
from .base_dataset import Dataset
from .base_raw_dataset import RawDataset
from .base_ml_task_dataset import MLTaskDataset, MLTaskPreparationConfig
from ..data_pipe_routines.missing_data_routines import BasicMissingDataHandler
from ..data_pipe_routines.data_type_routines import BasicFeatureTypeHandler
from ..utils import handle_targets
    

###################################################################################################################################
# Fetal CAD Dataset
###################################################################################################################################
class FetalCTGDataset(Dataset):

    def __init__(self):
        
        name = 'fetalctg'
        subject_area = 'Medical'
        year = 2019
        url = 'https://www.kaggle.com/datasets/akshat0007/fetalhr'
        download_link = None
        description = "This dataset explores the subjective quality assessment of digital colposcopies."
        notes = 'Extracted from Signals, CTG'
        data_type = 'numerical'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes
        )
        
        self.data_dir = os.path.join(DATA_DIR, self.name)
        self.raw_dataset: RawDataset = None
        self.ml_ready_dataset: MLReadyDataset = None
    
    def _load_raw_data(self) -> Tuple[pd.DataFrame, dict]:
        """
        Load raw dataset and specify meta data information
        
        Returns:
            raw_data: pd.DataFrame, raw data
            meta_data: dict, meta data
        """
        downloader = KaggleDownloader(
            dataset_name = 'akshat0007/fetalhr',
            file_names = ['CTG.csv'],
            download_all = True
        )
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'CTG.csv')).reset_index(drop=True)
        raw_data = raw_data.drop(columns = ['FileName', 'Date', 'SegFile'])
        raw_data = raw_data.dropna()
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        binary_features = [
            'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'
        ]
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = ['CLASS', 'NSP']
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + ordinal_features + multiclass_features
        ]
        
        target_features = [
            'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP',
            'CLASS', 'NSP'
        ]
        sensitive_features = []
        drop_features = []
        task_names = [
            'predict_pattern_A', 
            'predict_pattern_B', 
            'predict_pattern_C', 
            'predict_pattern_D', 
            'predict_pattern_E', 
            'predict_pattern_AD', 
            'predict_pattern_DE', 
            'predict_pattern_LD', 
            'predict_pattern_FS', 
            'predict_pattern_SUSP',
            'predict_pattern_CLASS',
            'predict_pattern_NSP'
        ]
        
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
        
        if drop_unused_targets is False:
            raise ValueError(f"drop_unused_targets is False, which is not supported for this dataset.")
               
        target_name = task_name.split('_')[-1]
        target_info = {
            'target': target_name,
            'task_type': 'classification'
        }
        
        data = handle_targets(data, raw_data_config, drop_unused_targets, target_info['target'])
        
        assert (target_info['target'] in data.columns), "Target feature not found in data."
        
        return data, target_info
    
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        
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


    
