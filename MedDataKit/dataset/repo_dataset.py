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
from ..downloader import URLDownloader
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
        source = 'uci'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            source = source
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
        task_names = ['Class']
        
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
        source = 'uci'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            source = source
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
        task_names = ['type_prediction', 'consensus_prediction']
        for i in range(6):
            task_names.append(f'expert_{i}_prediction')
        
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
        task_names = ['Diagnosis']
        
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
# Breast Cancer Wisconsin Dataset
###################################################################################################################################
class BreastCancerWisconsinDataset(Dataset):

    def __init__(self):
        
        name = 'breast_wisc'
        subject_area = 'Medical'
        year = 1995
        url = 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic'
        download_link = 'https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip'
        description = "Diagnostic Wisconsin Breast Cancer Database."
        notes = 'Extracted from Signals'
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
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'wdbc.data'), header = None, index_col = 0)
        raw_data.columns = ['diagnosis'] + [f'F{i-1}' for i in raw_data.columns[1:]]
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        binary_features = ['diagnosis']
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = []
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + ordinal_features + multiclass_features
        ]
        
        target_features = ['diagnosis']
        sensitive_features = []
        drop_features = []
        task_names = ['diagnosis']
        
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

        if task_name != 'diagnosis':
            raise ValueError(f"task_name is not 'diagnosis' for {self.name} dataset, which is not supported")
        
        target_info = {
            'target': 'diagnosis',
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
        
        name = 'dermatology'
        subject_area = 'Medical'
        year = 1997
        url = 'https://archive.ics.uci.edu/dataset/33/dermatology'
        download_link = 'https://archive.ics.uci.edu/static/public/33/dermatology.zip'
        description = "Aim for this dataset is to determine the type of Eryhemato-Squamous Disease."
        notes = ''
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
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'dermatology.data'), header=None, na_values='?')
        column_names = [
            'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules', 
            'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
            'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate',
            'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
            'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
            'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer',
            'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw_tooth_appearance_of_retes', 'follicular_horn_plug',
            'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate', 'band_like_infiltrate', 'Age', 'Class'
        ]
        raw_data.columns = column_names
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        binary_features = ['family_history']
        numerical_features = ['Age']
        ordinal_feature_order_dict = {}
        multiclass_features = ['Class']
        ordinal_features = [
            col for col in raw_data.columns 
            if col not in binary_features + numerical_features + multiclass_features
        ]
        for col in ordinal_features:
            ordinal_feature_order_dict[col] = [0, 1, 2, 3]
        
        target_features = ['Class']
        sensitive_features = []
        drop_features = []
        task_names = ['Class']
        
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

        if task_name != 'Class':
            raise ValueError(f"task_name is not 'Class' for {self.name} dataset, which is not supported")
        
        target_info = {
            'target': 'Class',
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
# Bone Transplant Dataset
###################################################################################################################################
class BoneTransplantDataset(Dataset):

    def __init__(self):
        
        name = 'bonetransplant'
        subject_area = 'Medical'
        year = 2020
        url = 'https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children?spm=wolai.workspace.0.0.5fd9761cERlBBA'
        download_link = 'https://archive.ics.uci.edu/static/public/565/bone+marrow+transplant+children.zip'
        description = "The data set describes pediatric patients with several hematologic diseases, " \
            "who were subject to the unmanipulated allogeneic unrelated donor hematopoietic stem cell transplantation."
        notes = ''
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
        data, meta = arff.loadarff(os.path.join(self.data_dir, 'bone-marrow.arff'))
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = df[col].replace(b'?', pd.NA)
            try:
                # Decode bytes and convert to numeric, keeping NaN values
                df[col] = pd.to_numeric(df[col].str.decode('utf-8'), errors='coerce')
            except (ValueError, AttributeError):
                continue
        df.replace('?', np.nan, inplace=True)
        df = df[df['ANCrecovery'] < 1000000]
        df = df.drop(columns = ['Disease'])
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
        numerical_features = [
            'Donorage', 'Recipientage', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'ANCrecovery', 
            'PLTrecovery', 'Rbodymass', 'time_to_aGvHD_III_IV', 'survival_time'
        ]
        multiclass_features = ['DonorABO', 'RecipientABO', 'CMVstatus', 'HLAmatch', 'Antigen', 'Alel', 'HLAgrI']
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['survival_status', 'survival_time']
        sensitive_features = ['Recipientgender']
        drop_features = []
        task_names = ['predict_survival', 'predict_survival_time']
        
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

        if task_name == 'predict_survival':
            from lifelines import KaplanMeierFitter
            fitter = KaplanMeierFitter()
            data['survival_status'] = data['survival_status'].astype(float)
            fitter.fit(data['survival_time'], data['survival_status'])
            target = fitter.predict(data['survival_time']).reset_index(drop = True)
            data['survival_risk'] = target
            data = data.drop(columns = ['survival_status', 'survival_time'])
            target_info = {
                'target': 'survival_risk',
                'task_type': 'regression'
            }
        elif task_name == 'predict_survival_time':
            target_info = {
                'target': 'survival_time',
                'task_type': 'regression'
            }
        else:
            raise ValueError(f"task_name {task_name} is not supported for {self.name} dataset")
        
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
            threshold1_num=0.5,
            threshold1_cat=0.5,
            threshold2_num=0.05,
            threshold2_cat=0.05
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        return data, missing_data_info
    
###################################################################################################################################
# Parkinsons Dataset
###################################################################################################################################
class ParkinsonsDataset(Dataset):

    def __init__(self):
        
        name = 'parkinsons'
        subject_area = 'Medical'
        year = 2020
        url = 'https://archive.ics.uci.edu/dataset/174/parkinsons'
        download_link = 'https://archive.ics.uci.edu/static/public/174/parkinsons.zip'
        description = "Oxford Parkinson's Disease Detection Dataset"
        notes = 'Extracted from Signal'
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
        raw_data = pd.read_csv(
            os.path.join(self.data_dir, 'parkinsons.data'), index_col = 0
        )
        raw_data.columns = [str(i) for i in raw_data.columns]

        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = []
        binary_features = ['status']
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['status']
        sensitive_features = []
        drop_features = []
        task_names = ['predict_status']
        
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

        if task_name == 'predict_status':
            target_info = {
                'target': 'status',
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
    

###################################################################################################
# Cervical Risk Dataset
###################################################################################################
class CervicalRiskDataset(Dataset):

    def __init__(self):
        
        name = 'cervical_risk'
        subject_area = 'Medical'
        year = 2023
        url = 'https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors?'
        download_link = 'https://archive.ics.uci.edu/static/public/383/cervical+cancer+risk+factors.zip'
        description = "This dataset focuses on the prediction of indicators/diagnosis of cervical cancer. " \
                      "The features cover demographic information, habits, and historic medical records."
        notes = ''
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
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'risk_factors_cervical_cancer.csv'), na_values='?')
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = []
        binary_features = ['status']
        numerical_features = [
            'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes (years)',
            'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
            'STDs: Number of diagnosis', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
        ]
        
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Biopsy']
        sensitive_features = ['Age']
        drop_features = []
        task_names = ['predict_biopsy']
        
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

        if task_name == 'predict_biopsy':
            target_info = {
                'target': 'Biopsy',
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
            threshold1_num = 0.5,
            threshold2_num = 0.05,
            threshold1_cat = 0.5,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info

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
        source = 'kaggle'
        
        super().__init__(
            name = name,
            description = description,
            collection_year = year,
            subject_area = subject_area,
            url = url,
            download_link = download_link,
            notes = notes,
            source = source
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
            'CLASS', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'NSP'
        ]
        sensitive_features = []
        drop_features = []
        task_names = [
            'predict_pattern_CLASS',
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