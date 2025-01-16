import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from MedDataKit.utils import column_check
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from MedDataKit.dataset.base_raw_dataset import RawDataset
from MedDataKit.dataset.base_ml_task_dataset import MLTaskDataset, MLTaskPreparationConfig
from MedDataKit.utils import update_feature_type

class Dataset(ABC):
    
    def __init__(
        self, 
        name: str, 
        description: str,
        collection_year: int = None,
        subject_area: str = None,
        url: str = None,
        download_link: str = None,
        notes: str = None,
        data_type: str = None
    ):
        
        # Dataset Information
        self.name = name
        self.description = description
        self.collection_year = collection_year
        self.subject_area = subject_area
        self.url = url
        self.download_link = download_link
        self.notes = notes
        self.data_type = data_type
        # Raw Dataset
        self.raw_dataset: RawDataset = None                    # raw dataset
        
        # ML Task Dataset
        self.ml_task_prep_config: MLTaskPreparationConfig = None              # ml task dataset configuration
        self.ml_task_dataset: MLTaskDataset = None            # ml task dataset
        
    @property
    def data_processing_config(self):
        return asdict(self.ml_task_config)
    
    def load_raw_data(self):
        """
        Load raw dataset and specify meta data information
        """
        raw_data = self._load_raw_data()
        meta_data = self._set_raw_data_config(raw_data)
        
        assert (
            'numerical_features' in meta_data and 
            'binary_features' in meta_data and 
            'multiclass_features' in meta_data and 
            'ordinal_features' in meta_data and 
            'ordinal_feature_order_dict' in meta_data
        ), "Meta data structure is not correct. numerical_features, binary_features," \
        "multiclass_features, ordinal_features, and ordinal_feature_order_dict must be specified in meta data"
        
        assert (
            'target_features' in meta_data and 
            'sensitive_features' in meta_data and 
            'drop_features' in meta_data and 
            'task_names' in meta_data
        ), "Meta data structure is not correct. target_features, sensitive_features, drop_features,"\
        "feature_groups, and task_names must be specified in meta data"
        
        assert (
            'feature_groups' in meta_data and 
            'fed_cols' in meta_data
        ), "Meta data structure is not correct. feature_groups and fed_cols must be specified in meta data"
        
        # Create raw dataset based on raw dataframe and meta data
        self.raw_dataset = RawDataset(raw_data, **meta_data)
        
        return self.raw_dataset.get_data()
    
    def get_task_names(self):
        """
        Get all task names
        """
        try:
            return self.raw_dataset.task_names
        except:
            raise ValueError("Raw dataset is not loaded")
    
    def generate_ml_task_dataset(
        self, task_name: str = None, config: dict = None, verbose: bool = False
    ) -> pd.DataFrame:
        """
        Generate ml-ready data for a specific task and specify corresponding meta data information
        """
        
        ##########################################################################################################
        # ML task preparation configuration
        if config is None:
            config = {}
        
        self.ml_task_prep_config = MLTaskPreparationConfig(**config)
        
        if verbose:
            print(asdict(self.ml_task_prep_config))
        
        ##########################################################################################################
        # Get copy of raw data
        raw_data = self.raw_dataset.get_data().copy()
        raw_data, ordinal_feature_codes = self.raw_dataset.factorize_ordinal_features(
            raw_data, self.raw_dataset.ordinal_features
        )
        raw_data_config = self.raw_dataset.raw_data_config
        
        if verbose:
            print("Raw data shape: ", raw_data.shape)
        
        ##########################################################################################################
        # Set target variable based on task name
        if task_name is None:
            task_name = raw_data_config['task_names'][0]
        
        drop_unused_targets = self.ml_task_prep_config.drop_unused_targets
        raw_data, target_info = self._set_target_feature(
            raw_data, raw_data_config, task_name, drop_unused_targets
        )
        
        if verbose:
            print('After setting target feature: ', raw_data.shape)
        
        assert (
            'target' in target_info and 
            'task_type' in target_info and
             target_info['task_type'] in ['classification', 'regression', 'survival']
        ), "Target information is not correct. target, task_type must be specified in target_info"
        
        raw_data_config.update(target_info)
        
        ##########################################################################################################
        # Feature Type Specification and Engineering
        ml_data, feature_config = self._feature_engineering(raw_data, raw_data_config, self.ml_task_prep_config)
        
        assert (
            'numerical_features' in feature_config and
            'categorical_features' in feature_config
        ), "Feature engineering configuration is not correct. " \
            "numerical_features, categorical_features, " \
            "must be specified in feature_config"
        
        # update feature type
        target_feature = target_info['target']
        numerical_features = feature_config['numerical_features']
        categorical_features = feature_config['categorical_features']
        numerical_features, categorical_features = update_feature_type(
            ml_data, numerical_features, categorical_features, target_feature
        )
        
        if verbose:
            print('After feature engineering: ', ml_data.shape)
        
        ##########################################################################################################
        # Handle missing data
        missing_strategy = self.ml_task_prep_config.missing_strategy
        missing_drop_thres = self.ml_task_prep_config.missing_drop_thres
        
        ml_data, missing_data_info = self.handle_missing_data(
            ml_data, missing_strategy, missing_drop_thres, categorical_features
        )
        # update feature type
        numerical_features, categorical_features = update_feature_type(
            ml_data, numerical_features, categorical_features, target_feature
        )
        
        if verbose:
            print('After handling missing data: ', ml_data.shape)
        
        ##########################################################################################################
        # Create ML task dataset
        ml_data_config = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'target': target_feature,
            'task_type': target_info['task_type'],
        }
        
        self.ml_task_dataset = MLTaskDataset(
            ml_data, 
            task_name,
            numerical_encoding=self.ml_task_prep_config.numerical_encoding,
            categorical_encoding=self.ml_task_prep_config.categorical_encoding,
            **ml_data_config
        )
        
        if verbose:
            print('Final ml task dataset shape: ', self.ml_task_dataset.data.shape)

        return self.ml_task_dataset.data
    
    def handle_missing_data(
        self, 
        data: pd.DataFrame, 
        missing_strategy: str,
        missing_drop_thres: float,
        categorical_features: list
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Handle missing data
        
        Args:
            data: pd.DataFrame, raw data
            missing_strategy: str, missing data handlingstrategy
            categorical_features: list, categorical features
            
        Returns:
            data: pd.DataFrame, processed data
            missing_data_info: dict, missing data processing information
        """
        
        data = data.fillna(np.nan)
        
        # Drop features with missing ratio greater than missing_drop_thres
        for col in data.columns:
            missing_ratio = data[col].isna().sum() / len(data)
            if missing_ratio > missing_drop_thres:
                data = data.drop(columns=[col])
        
        # Handle missing data for the remaining features
        if missing_strategy == 'keep':
            missing_data_info = {}
        elif missing_strategy == 'drop':
            data = data.dropna()
            missing_data_info = {}
        elif missing_strategy == 'impute':
            data, missing_data_info = self._handle_missing_data(data, categorical_features)
        elif missing_strategy == 'impute_cat':
            data_cat = data[categorical_features].copy().reset_index(drop=True)
            data_num = data.drop(columns=categorical_features).reset_index(drop=True)
            data_cat, missing_data_info = self._handle_missing_data(data_cat, categorical_features)
            data = pd.concat([data_num, data_cat], axis=1)
        else:
            raise ValueError("Invalid missing strategy")
        
        return data, missing_data_info
    
    def show_dataset_info(self):
        """
        Show all information of raw dataset
        """
        print("=========================================================")
        print(f"Dataset name: {self.name} ({self.collection_year}) Subject Area: {self.subject_area}")
        print(f"Dataset URL: {self.url}")
        print(f"Dataset description: {self.description}")
        print(f"Dataset notes: {self.notes}")
        print(f"Dataset data type: {self.data_type}")
        print("=========================================================")
        self.raw_dataset.show_dataset_info()
        print("=========================================================")
        self.raw_dataset.show_missing_data_statistics()
        print("=========================================================")
    
    ##########################################################################################################
    # Abstract methods
    ##########################################################################################################
    @abstractmethod
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset
        
        Returns:
            raw_data: pd.DataFrame, raw data
        """
        return pd.DataFrame()
    
    @abstractmethod
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        return {}
    
    @abstractmethod
    def _set_target_feature(
        self, 
        data: pd.DataFrame, 
        raw_data_config: dict, 
        task_name: str,
        drop_unused_targets: bool
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
        return data, {}
    
    @abstractmethod
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generate ml-ready data for a specific task and specify corresponding meta data information
        1. feature selection:
            - drop features with are useless for the task (e.g. id, index, description, etc.)
        2. specify numerical and categorical features
            - numerical features: numerical + ordinal or only numerical
            - categorical features: binary + multiclass + ordinal or binary + multiclass
        3. optional:
            - outlier remover
            - gaussianization
            - feature engineering
        """
        return data, {}
    
    @abstractmethod
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
        return data, {}
            

