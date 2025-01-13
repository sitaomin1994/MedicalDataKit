import pandas as pd
import os
import pyreadr
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from config import DATA_DIR, DATA_DOWNLOAD_DIR

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

###################################################################################################################################
# Arrhythmia Dataset
###################################################################################################################################
class ArrhythmiaDataset(Dataset):

    def __init__(self):
        
        name = 'arrhythmia'
        subject_area = 'Medical'
        year = 1997
        url = 'https://archive.ics.uci.edu/dataset/5/arrhythmia'
        download_link = 'https://archive.ics.uci.edu/static/public/5/arrhythmia.zip'
        description = "Distinguish between the presence and absence of cardiac arrhythmia and classify" \
                      "it in one of the 16 groups."
        notes = 'Extracted from Signals'
        
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
        
    def _load_raw_data(self):
        
        # download data
        downloader = UCIMLDownloader(url = self.download_link)
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(
            os.path.join(self.data_dir, 'arrhythmia.data'), header = None, na_values='?'
        )
        
        columns = [
            'Age', 'Sex', 'Height', 'Weight', 'QRS duration', 'P-R interval', 'Q-T interval', 
            'T interval', 'P interval', 'QRS_angle', 'T_angle', 'P_angle', 'QRST_angle', 'J_angle', 'Heart rate'
        ]
        
        for key in ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            columns.append(f'{key}_Q')
            columns.append(f'{key}_R')
            columns.append(f'{key}_S')
            columns.append(f'{key}_R2')
            columns.append(f'{key}_S2')
            columns.append(f'{key}_num_deflections')
            columns.append(f'{key}_Ragged_R')
            columns.append(f'{key}_Diphasic_R')
            columns.append(f'{key}_Ragged_P')
            columns.append(f'{key}_Diphasic_P')
            columns.append(f'{key}_Ragged_T')
            columns.append(f'{key}_Diphasic_T')
        
        for key in ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            columns.append(f'{key}_Amp_JJ')
            columns.append(f'{key}_Amp_Q')
            columns.append(f'{key}_Amp_R')
            columns.append(f'{key}_Amp_S')
            columns.append(f'{key}_Amp_R2')
            columns.append(f'{key}_Amp_S2')
            columns.append(f'{key}_Amp_P')
            columns.append(f'{key}_Amp_T')
            columns.append(f'{key}_Amp_QRSA')
            columns.append(f'{key}_Amp_QRSTA')
        
        columns.append('Class')
        raw_data.columns = columns
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
    
        # Specify meta data
        binary_features = ['Sex']
        for key in ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            binary_features.append(f'{key}_Ragged_R')
            binary_features.append(f'{key}_Diphasic_R')
            binary_features.append(f'{key}_Ragged_P')
            binary_features.append(f'{key}_Diphasic_P')
            binary_features.append(f'{key}_Ragged_T')
            binary_features.append(f'{key}_Diphasic_T')
        multiclass_features = ['Class']
        ordinal_features = []
        ordinal_feature_order_dict = {}
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Class']
        sensitive_features = ['Age', 'Sex']
        drop_features = []
        feature_groups = {}
        task_names = ['Class']
        
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
            'task_names': task_names
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
        if task_name == 'Class':
            
            target_info = {
                'target': 'Class',
                'task_type': 'classification'
            }
            
        else:
            raise ValueError(f"Invalid task name: {task_name}")
        
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
        missing_data_handler = BasicMissingDataHandler(
            threshold1_num = 0.5, threshold2_num = 0.2, 
            threshold1_cat = 0.5, threshold2_cat = 0.2
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        assert data.isna().sum().sum() == 0, "Missing data is not handled"
        return data, missing_data_info
    

###################################################################################################################################
# Colposcopy Dataset
###################################################################################################################################
class ColposcopyDataset(Dataset):

    def __init__(self):
        
        name = 'colposcopy'
        subject_area = 'Medical'
        year = 2017
        url = 'https://archive.ics.uci.edu/dataset/384/quality+assessment+of+digital+colposcopies'
        download_link = 'https://archive.ics.uci.edu/static/public/384/quality+assessment+of+digital+colposcopies.zip'
        description = "This dataset explores the subjective quality assessment of digital colposcopies."
        notes = 'Extracted from Signals'
        
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
        downloader = UCIMLDownloader(url = self.download_link)
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        file_dir = os.path.join(self.data_dir, 'Quality Assessment - Digital Colposcopy')
        green = pd.read_csv(os.path.join(file_dir, 'green.csv'))
        green['class'] = 'green'
        hinselmann = pd.read_csv(os.path.join(file_dir, 'hinselmann.csv'))
        hinselmann['class'] = 'hinselmann'
        schiller = pd.read_csv(os.path.join(file_dir, 'schiller.csv'))
        schiller['class'] = 'schiller'
        
        raw_data = pd.concat([green, hinselmann, schiller], axis=0)
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        binary_features = [
            'experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5', 'consensus'
        ]
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = ['class']
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + ordinal_features + multiclass_features
        ]
        
        target_features = [
            'class', 'experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5', 'consensus'
        ]
        sensitive_features = []
        drop_features = []
        feature_groups = {}
        task_names = ['type_prediction', 'consensus_prediction']
        for i in range(6):
            task_names.append(f'expert_{i}_prediction')
        
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
            'ordinal_feature_order_dict': ordinal_feature_order_dict
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
        
        if task_name == 'type_prediction':
            target_info = {
                'target': 'class',
                'task_type': 'classification'
            }
            
        elif task_name == 'consensus_prediction':
            target_info = {
                'target': 'consensus',
                'task_type': 'classification'
            }
        else:
            expert_num = int(task_name.split('_')[1])
            target_info = {
                'target': f'experts::{expert_num}',
                'task_type': 'classification'
            }
        
        data = handle_targets(data, raw_data_config, drop_unused_targets, target_info['target'])
        
        assert (target_info['target'] in data.columns), "Target feature not found in data"
        
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

###################################################################################################################################
# Dermatology Dataset
###################################################################################################################################
class ZAlizadehsaniDataset(Dataset):

    def __init__(self):
        
        name = 'zalizadehsani'
        subject_area = 'Medical'
        year = 2017
        url = 'https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset'
        download_link = 'https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip'
        description = "Collections for CAD diagnosis."
        notes = 'Extracted from Signals, CAD'
        
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
        raw_data = pd.read_excel(
            os.path.join(self.data_dir, 'extention of Z-Alizadeh sani dataset.xlsx')
        )
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        numerical_features = [
            'Age', 'Weight', 'Length', 'BMI', 'BP', 'PR', 'FBS', 'CR',
            'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph',
            'Neut', 'PLT', 'EF-TTE',
        ]
        
        ordinal_features = []
        ordinal_feature_order_dict = {}
        
        multiclass_features = [
            'BBB', 'VHD', 'Region RWMA'
        ]
        
        target_features = [
            'LAD', 'LCX', 'RCA', 'Cath'
        ]
        
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + ordinal_features + multiclass_features
        ]
        
        sensitive_features = ['Age', 'Sex']
        drop_features = []
        feature_groups = {}
        task_names = ['LAD', 'LCX', 'RCA', 'Cath']
        
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
            'ordinal_feature_order_dict': ordinal_feature_order_dict
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
            raise ValueError(f"drop_unused_targets is False for {self.name} dataset, which is not supported")

        target_features = raw_data_config['target_features']
        if task_name == 'Cath':
            target_info = {
                'target': 'Cath',
                'task_type': 'classification'
            }
        elif task_name == 'LAD':
            target_info = {
                'target': 'LAD',
                'task_type': 'classification'
            }
        elif task_name == 'LCX':
            target_info = {
                'target': 'LCX',
                'task_type': 'classification'
            }
        elif task_name == 'RCA':
            target_info = {
                'target': 'RCA',
                'task_type': 'classification'
            }
        
        data = handle_targets(data, raw_data_config, drop_unused_targets, target_info['target'])
        
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
# SPECTF Dataset
###################################################################################################################################
class SPECTFDataset(Dataset):

    def __init__(self):
        
        name = 'spectf'
        subject_area = 'Medical'
        year = 2001
        url = 'https://archive.ics.uci.edu/dataset/96/spectf+heart'
        download_link = 'https://archive.ics.uci.edu/static/public/96/spectf+heart.zip'
        description = "Data on cardiac Single Proton Emission Computed Tomography (SPECT) images." \
                      "Each patient classified into two categories: normal and abnormal."
        notes = 'Extracted from Signals'
        data_type = 'numerical'
        
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
        downloader = UCIMLDownloader(url = self.download_link)
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        train = pd.read_csv(os.path.join(self.data_dir, 'SPECTF.train'), header = None).reset_index(drop = True)
        test = pd.read_csv(os.path.join(self.data_dir, 'SPECTF.test'), header = None).reset_index(drop = True)
        raw_data = pd.concat([train, test], axis = 0)
        
        columns = ['Diagnosis']
        for i in range(1, 23):
            columns.append(f'F{i}R')
            columns.append(f'F{i}S')
        raw_data.columns = columns
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        binary_features = ['Diagnosis']
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = []
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + ordinal_features + multiclass_features
        ]
        
        target_features = ['Diagnosis']
        sensitive_features = []
        drop_features = []
        feature_groups = {}
        task_names = ['Diagnosis']
        
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
            'ordinal_feature_order_dict': ordinal_feature_order_dict
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

        if task_name != 'Diagnosis':
            raise ValueError(f"task_name is not 'Diagnosis' for {self.name} dataset, which is not supported")
        
        target_info = {
            'target': 'Diagnosis',
            'task_type': 'classification'
        }
        
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
# Dermatology Dataset
###################################################################################################################################
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



    
