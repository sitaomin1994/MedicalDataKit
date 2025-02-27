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
# Zigong Heart Failure Dataset
###################################################################################################################################
class ZigongHeartFailureDataset(Dataset):
    
    def __init__(self):
        
        name = 'zigongheartfailure'
        subject_area = 'Medical'
        year = 2020
        url = 'https://physionet.org/content/heart-failure-zigong/1.3/'
        download_link = 'https://physionet.org/content/heart-failure-zigong/1.3/'
        description = "Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data"
        notes = 'EHR'
        data_type = 'mixed'
        self.pub_link = 'Electronic healthcare recordsand external outcome data forhospitalized patients with heartfailure'
        source = 'physionet'
        
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
        if (
            not os.path.exists(os.path.join(self.data_dir, 'dat.csv')) and
            not os.path.exists(os.path.join(self.data_dir, 'dat_md.csv'))
        ):
            raise Exception(f'Data {self.name} does not exist in {self.data_dir}, please add the data first')
        
        # load raw data
        data = pd.read_csv(os.path.join(self.data_dir, 'dat.csv'), index_col=0)
        data_md = pd.read_csv(os.path.join(self.data_dir, 'dat_md.csv'), index_col=0)
        
        # long to wide drug
        data_md['has_drug'] = 1
        data_md_wide = data_md.pivot(index="inpatient.number", columns="Drug_name").reset_index()
        data_md_wide.columns = ["inpatient.number"] + [f"Drug_{i}" for i in range(1, data_md_wide.shape[1])]
        data_md_wide = data_md_wide.fillna(0)
        data = data.merge(data_md_wide, on='inpatient.number')
        data = data.drop(columns=['inpatient.number'])
        
        return data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = ['eye.opening', 'verbal.response', 'movement']
        ordinal_feature_order_dict = {
            'eye.opening': [1, 2, 3, 4],
            'verbal.response': [1, 2, 3, 4, 5],
            'movement': [1, 2, 3, 4, 5],
        }
        binary_features = [
            'gender', 'admission.way',
            'myocardial.infarction', 'congestive.heart.failure', 'peripheral.vascular.disease', 'cerebrovascular.disease',
            'dementia', 'Chronic.obstructive.pulmonary.disease', 'connective.tissue.disease', 'peptic.ulcer.disease',
            'diabetes', 'moderate.to.severe.chronic.kidney.disease', 'hemiplegia', 'leukemia', 'malignant.lymphoma',
            'solid.tumor', 'liver.disease', 'AIDS', 'type.II.respiratory.failure', 'oxygen.inhalation', 'acute.renal.failure',
            'death.within.28.days', 're.admission.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 
            'death.within.6.months', 're.admission.within.6.months', 'return.to.emergency.department.within.6.months',
        ]
        
        binary_features.extend(
            [f'Drug_{i}' for i in range(1, 26)]
        )

        multiclass_features = [
            'DestinationDischarge', 'discharge.department', 'occupation', 'admission.ward', 'type.of.heart.failure',
            'NYHA.cardiac.function.classification', 'consciousness', 'Killip.grade',
            'respiratory.support.', 'outcome.during.hospitalization', 'ageCat'
        ]

        numerical_features = [
            col for col in raw_data.columns if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = [
            'outcome.during.hospitalization', 
            'death.within.28.days', 
            'death.within.3.months', 
            'death.within.6.months', 
            're.admission.within.28.days', 
            're.admission.within.3.months',
            're.admission.within.6.months', 
            'return.to.emergency.department.within.6.months',
        ]
        sensitive_features = ['gender', 'ageCat']
        drop_features = []
        task_names = [
            'predict_outcome',
            'predict_death_30',
            'predict_death_90',
            'predict_death_180',
            'predict_readmission_30',
            'predict_readmission_90',
            'predict_readmission_180',
            'predict_readmission_ed',
        ]
        
        feature_groups = {
            'demongraphic': [
                'DestinationDischarge', 'admission.ward', 'admission.way', 'occupation', 'discharge.department', 
                'visit.times', 'gender', 'weight', 'height', 'BMI', 'ageCat',
                'outcome.during.hospitalization', 'death.within.28.days', 'death.within.3.months', 
                'death.within.6.months', 're.admission.within.28.days', 're.admission.within.3.months', 
                're.admission.within.6.months', 'return.to.emergency.department.within.6.months',
                'dischargeDay', 'time.of.death..days.from.admission.', 
                're.admission.time..days.from.admission.', 'time.to.emergency.department.within.6.months'
            ],
            'clinical_status': [
                'body.temperature', 'pulse', 'respiration', 'systolic.blood.pressure', 'diastolic.blood.pressure', 
                'map', 'NYHA.cardiac.function.classification', 'Killip.grade', 'GCS', 'LVEF', 
                'left.ventricular.end.diastolic.diameter.LV', 'mitral.valve.EMS', 'mitral.valve.AMS', 
                'EA', 'tricuspid.valve.return.velocity', 'tricuspid.valve.return.pressure', 
                'type.II.respiratory.failure', 'consciousness', 'eye.opening', 'verbal.response', 
                'movement', 'respiratory.support.', 'type.of.heart.failure'
            ],
            'comorbidity': [
                'myocardial.infarction', 'congestive.heart.failure', 'peripheral.vascular.disease', 
                'cerebrovascular.disease', 'dementia', 'Chronic.obstructive.pulmonary.disease', 
                'connective.tissue.disease', 'peptic.ulcer.disease', 'diabetes', 
                'moderate.to.severe.chronic.kidney.disease', 'hemiplegia', 'leukemia', 'malignant.lymphoma', 
                'solid.tumor', 'liver.disease', 'AIDS', 'CCI.score'
            ],
            'laboratory': [
                'oxygen.inhalation', 'fio2', 'acute.renal.failure', 
                'creatinine.enzymatic.method', 'urea', 'uric.acid', 'glomerular.filtration.rate', 'cystatin', 
                'white.blood.cell', 'monocyte.ratio', 'monocyte.count', 'red.blood.cell', 
                'coefficient.of.variation.of.red.blood.cell.distribution.width', 
                'standard.deviation.of.red.blood.cell.distribution.width', 'mean.corpuscular.volume', 
                'hematocrit', 'lymphocyte.count', 'mean.hemoglobin.volume', 'mean.hemoglobin.concentration', 
                'mean.platelet.volume', 'basophil.ratio', 'basophil.count', 'eosinophil.ratio', 'eosinophil.count', 
                'hemoglobin', 'platelet', 'platelet.distribution.width', 'platelet.hematocrit', 'neutrophil.ratio', 
                'neutrophil.count', 'D.dimer', 'international.normalized.ratio', 'activated.partial.thromboplastin.time', 
                'thrombin.time', 'prothrombin.activity', 'prothrombin.time.ratio', 'fibrinogen', 'high.sensitivity.troponin', 
                'myoglobin', 'carbon.dioxide.binding.capacity', 'calcium', 'potassium', 'chloride', 'sodium', 
                'Inorganic.Phosphorus', 'serum.magnesium', 'creatine.kinase.isoenzyme.to.creatine.kinase', 
                'hydroxybutyrate.dehydrogenase.to.lactate.dehydrogenase', 'hydroxybutyrate.dehydrogenase', 
                'glutamic.oxaloacetic.transaminase', 'creatine.kinase', 'creatine.kinase.isoenzyme', 
                'lactate.dehydrogenase', 'brain.natriuretic.peptide', 'high.sensitivity.protein', 
                'nucleotidase', 'fucosidase', 'albumin', 'white.globulin.ratio', 'cholinesterase', 
                'glutamyltranspeptidase', 'glutamic.pyruvic.transaminase', 'glutamic.oxaliplatin', 
                'indirect.bilirubin', 'alkaline.phosphatase', 'globulin', 'direct.bilirubin', 'total.bilirubin', 
                'total.bile.acid', 'total.protein', 'erythrocyte.sedimentation.rate', 'cholesterol', 
                'low.density.lipoprotein.cholesterol', 'triglyceride', 'high.density.lipoprotein.cholesterol', 
                'homocysteine', 'apolipoprotein.A', 'apolipoprotein.B', 'lipoprotein', 'pH', 'standard.residual.base', 
                'standard.bicarbonate', 'partial.pressure.of.carbon.dioxide', 'total.carbon.dioxide', 'methemoglobin', 
                'hematocrit.blood.gas', 'reduced.hemoglobin', 'potassium.ion', 'chloride.ion', 'sodium.ion', 
                'glucose.blood.gas', 'lactate', 'measured.residual.base', 'measured.bicarbonate', 'carboxyhemoglobin', 
                'body.temperature.blood.gas', 'oxygen.saturation', 'partial.oxygen.pressure', 'oxyhemoglobin', 
                'anion.gap', 'free.calcium', 'total.hemoglobin'
            ],
            'drugs': [
                'Drug_1', 'Drug_2', 'Drug_3', 'Drug_4', 'Drug_5', 'Drug_6', 'Drug_7', 'Drug_8', 'Drug_9', 
                'Drug_10', 'Drug_11', 'Drug_12', 'Drug_13', 'Drug_14', 'Drug_15', 'Drug_16', 'Drug_17', 
                'Drug_18', 'Drug_19', 'Drug_20', 'Drug_21', 'Drug_22', 'Drug_23', 'Drug_24', 'Drug_25'
            ]
        }
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
        if task_name == 'predict_outcome':
            target_info = {
                'target': 'outcome.during.hospitalization',
                'task_type': 'classification'
            }
        elif task_name == 'predict_death_30':
            target_info = {
                'target': 'death.within.28.days',
                'task_type': 'classification'
            }
        elif task_name == 'predict_death_90':
            target_info = {
                'target': 'death.within.3.months',
                'task_type': 'classification'
            }
        elif task_name == 'predict_death_180':
            target_info = {
                'target': 'death.within.6.months',
                'task_type': 'classification'
            }
        elif task_name == 'predict_readmission_30':
            target_info = {
                'target': 're.admission.within.28.days',
                'task_type': 'classification'
            }
        elif task_name == 'predict_readmission_90':
            target_info = {
                'target': 're.admission.within.3.months',
                'task_type': 'classification'
            }
        elif task_name == 'predict_readmission_180':
            target_info = {
                'target': 're.admission.within.6.months',
                'task_type': 'classification'
            }
        elif task_name == 'predict_readmission_ed':
            target_info = {
                'target': 'return.to.emergency.department.within.6.months',
                'task_type': 'classification'
            }
        else:
            raise ValueError(f"task name {task_name} is not supported")
        
        target_features = raw_data_config['target_features']
        
        if drop_unused_targets is False:
            drop_unused_targets = True
            print(f"Warning: drop_unused_targets is set to True for this dataset")
        
        data = handle_targets(data, raw_data_config, drop_unused_targets, target_info['target'])
        
        assert target_info['target'] in data.columns, f"target {target_info['target']} is not in data columns"
        
        return data, target_info
    
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Feature engineering
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """
        
        # drop features
        data = data.drop(columns=[
            'time.of.death..days.from.admission.', 
            're.admission.time..days.from.admission.', 
            'time.to.emergency.department.within.6.months',
            'dischargeDay',
            'DestinationDischarge', 
            'discharge.department'
        ])
        
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
            threshold1_num = 0.55,
            threshold2_num = 0.05,
            threshold1_cat = 0.5,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info
    
    
###################################################################################################################################
# MIMICII IACCD
###################################################################################################################################
class MIMIC2IACDataset(Dataset):

    def __init__(self):
        
        name = 'mimicii_iac'
        subject_area = 'Medical'
        year = 2008
        url = 'https://physionet.org/content/mimic2-iaccd/1.0/'
        download_link = 'https://physionet.org/content/mimic2-iaccd/1.0/'
        description = "Clinical data from the MIMIC-II database for a case study on indwelling arterial catheters"
        notes = 'MIMIC-II'
        data_type = 'mixed'
        self.pub_link = ''
        source = 'physionet'
        
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
        if not os.path.exists(os.path.join(self.data_dir, 'full_cohort_data.csv')):
            raise Exception(f'Data {self.name} does not exist in {self.data_dir}, please add the data first')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'full_cohort_data.csv'))
        
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
            'icu_los_day', 'age', 'weight_first', 'bmi', 'sapsi_first', 'sofa_first', 
            'day_icu_intime_num', 'hour_icu_intime', 'mort_day_censored', 'map_1st', 'hr_1st', 'temp_1st', 
            'spo2_1st', 'abg_count', 'wbc_first', 'hgb_first', 'platelet_first', 'sodium_first', 'potassium_first',
            'tco2_first', 'chloride_first', 'bun_first', 'creatinine_first', 'po2_first', 'pco2_first', 'iv_day_1',
            'hospital_los_day'
        ]
        multiclass_features = [
            'service_unit', 'day_icu_intime'
        ]
        
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = [
            'hosp_exp_flg', 'icu_exp_flg', 'icu_los_day', 'hospital_los_day', 'censor_flg', 'day_28_flg'
        ]
        sensitive_features = ['age']
        drop_features = []
        task_names = [
            'predict_censor',
            'predict_hosp_expire',
            'predict_day_28',  
            'predict_icu_expire', 
            'predict_icu_los', 
            'predict_hospital_los'
        ]
        
        feature_groups = {
            'demongraphic': [
                'age', 'gender_num', 'weight_first', 'bmi', 'sapsi_first', 'sofa_first', 'service_unit', 'service_num', 
                'day_icu_intime', 'day_icu_intime_num', 'hour_icu_intime', 'day_28_flg', 'aline_flg', 'mort_day_censored',
                'hosp_exp_flg', 'icu_exp_flg', 'icu_los_day', 'hospital_los_day'
            ],
            'comborbidity': [
                'censor_flg', 'sepsis_flg', 'chf_flg', 'afib_flg', 'renal_flg', 'liver_flg', 'copd_flg', 
                'cad_flg', 'stroke_flg', 'mal_flg', 'resp_flg'
            ],
            'clinical_status': [
                'map_1st', 'hr_1st', 'temp_1st', 'spo2_1st', 'abg_count', 'wbc_first', 'hgb_first', 
                'platelet_first', 'sodium_first', 'potassium_first', 'tco2_first', 'chloride_first', 
                'bun_first', 'creatinine_first', 'po2_first', 'pco2_first', 'iv_day_1'
            ]
        }
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

        if task_name == 'predict_hosp_expire':
            target_info = {
                'target': 'hosp_exp_flg',
                'task_type': 'classification'
            }
            data = data.drop(columns=['icu_exp_flg', 'day_28_flg', 'censor_flg', 'hospital_los_day', 'icu_los_day'])
        elif task_name == 'predict_icu_expire':
            target_info = {
                'target': 'icu_exp_flg',
                'task_type': 'classification'
            }
            data = data.drop(columns=['hosp_exp_flg', 'day_28_flg', 'censor_flg', 'hospital_los_day', 'icu_los_day'])
        elif task_name == 'predict_icu_los':
            target_info = {
                'target': 'icu_los_day',
                'task_type': 'regression'
            }
            data = data.drop(columns=['hosp_exp_flg', 'icu_exp_flg', 'hospital_los_day', 'day_28_flg', 'censor_flg'])
        elif task_name == 'predict_hospital_los':
            target_info = {
                'target': 'hospital_los_day',
                'task_type': 'regression'
            }
            data = data.drop(columns=['hosp_exp_flg', 'icu_exp_flg', 'icu_los_day', 'day_28_flg', 'censor_flg'])
        elif task_name == 'predict_day_28':
            target_info = {
                'target': 'day_28_flg',
                'task_type': 'classification'
            }
            data = data.drop(columns=['hosp_exp_flg', 'icu_exp_flg', 'icu_los_day', 'hospital_los_day', 'censor_flg'])
        elif task_name == 'predict_censor':
            target_info = {
                'target': 'censor_flg',
                'task_type': 'classification'
            }
            data = data.drop(columns=['hosp_exp_flg', 'icu_exp_flg', 'icu_los_day', 'hospital_los_day', 'day_28_flg'])
        else:
            raise ValueError(f"task name {task_name} is not supported")
        
        assert target_info['target'] in data.columns, f"target {target_info['target']} is not in data columns"
        
        return data, target_info
    
    def _feature_engineering(
        self, data: pd.DataFrame, data_config: dict, ml_task_prep_config: MLTaskPreparationConfig = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Feature engineering
        
        Args:
            data: pd.DataFrame, raw data
            raw_data_config: dict, raw data configuration {'target', 'task_type'}
            task_name: str, task name
            
        Returns:
            data: pd.DataFrame, processed data
            target_info: dict, target information
        """
        
        # drop features
        drop_cols = ['day_icu_intime', 'mort_day_censored']
        data = data.drop(columns=drop_cols)
        
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
            threshold1_cat = 0.7,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info

###################################################################################################################################
# Myocardial infarction complications Dataset
###################################################################################################################################
class MIComplicationsDataset(Dataset):

    def __init__(self):
        
        name = 'micomplications'
        subject_area = 'Medical'
        year = 2024
        url = 'https://figshare.le.ac.uk/articles/dataset/Myocardial_infarction_complications_Database/12045261'
        download_link = 'https://archive.ics.uci.edu/static/public/579/myocardial+infarction+complications.zip'
        description = "Prediction of myocardial infarction complications from lancester medical center"
        notes = 'Clinical'
        data_type = 'mixed'
        self.pub_link = ''
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
        if not os.path.exists(os.path.join(self.data_dir, 'MI.data')):
            downloader = URLDownloader(url = self.download_link, zipfile = True)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(
            os.path.join(self.data_dir, 'MI.data'), header=None, index_col = 0, na_values = ['?']
        )
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
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = [
            'STENOK_AN', 'DLIT_AG',  'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n',
            'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n'
        ]
        ordinal_feature_order_dict = {
            'STENOK_AN': [0, 1, 2, 3, 4, 5, 6],
            'DLIT_AG': [0, 1, 2, 3, 4, 5, 6, 7],
            'TIME_B_S': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'R_AB_1_n': [0, 1, 2, 3],
            'R_AB_2_n': [0, 1, 2, 3],
            'R_AB_3_n': [0, 1, 2, 3],
            'NA_R_1_n': [0, 1, 2, 3, 4], 
            'NA_R_2_n': [0, 1, 2, 3],
            'NA_R_3_n': [0, 1, 2],
            'NOT_NA_1_n': [0, 1, 2, 3, 4],
            'NOT_NA_2_n': [0, 1, 2, 3],
            'NOT_NA_3_n': [0, 1, 2]
            
        }
        binary_features = [
            'SEX',
            'IBS_NASL', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08',
            'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10',
            'endocr01', 'endocr02', 'endocr03', 
            'zableg01', 'zableg02', 'zableg03', 'zableg04', 'zableg06',
            'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'IM_PG_P', 
            'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08',
            'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06',
            'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 
            'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07',
            'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12',
            'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08',
            'GIPO_K', 'GIPER_Na',
            'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 
            'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n',
             # target
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV',
            'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN'
        ]
        multiclass_features = [
            'FK_STENOK', 'IBS_POST', 'GB', 'SIM_GIPERT', 'ZSN_A', 'ant_im', 'lat_im', 'inf_im', 'post_im', 
            # target
            'LET_IS'
        ]
        
        numerical_features = [
            'AGE', 'INF_ANAM', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 
            'K_BLOOD',  'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 
        ]
        
        target_features = [
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV',
            'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN',  'LET_IS'
        ]
        sensitive_features = ['AGE', 'SEX']
        drop_features = []
        task_names = [
            'predict_lethal_binary', 'predict_lethal', 'predict_heart_failure', 'predict_num_complications'
        ]
        
        feature_groups = {
            'patient_info': [
                'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 
                'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 
                'endocr01', 'endocr02', 'endocr03', 'zableg01', 'zableg02', 'zableg03', 'zableg04', 'zableg06',
                'LET_IS', 'A_V_BLOK', 'JELUD_TAH', 'P_IM_STEN', 'REC_IM', 'PREDS_TAH', 'RAZRIV', 'DRESSLER', 'FIBR_JELUD', 
                'FIBR_PREDS', 'ZSN', 'OTEK_LANC'
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
        target_features = raw_data_config['target_features']
        if task_name == 'predict_lethal':
            target_info = {
                'target': 'LET_IS',
                'task_type': 'classification'
            }
            target_features.remove('LET_IS')
            data = data.drop(columns=target_features)
        
        elif task_name == 'predict_lethal_binary':
            data['LET_IS_binary'] = data['LET_IS'].apply(lambda x: 1 if x == '0' else 0)
            
            target_info = {
                'target': 'LET_IS_binary',
                'task_type': 'classification'
            }
            data = data.drop(columns=target_features)
        
        elif task_name == 'predict_heart_failure':
            target_info = {
                'target': 'ZSN',
                'task_type': 'classification'
            }
            target_features.remove('ZSN')
            data = data.drop(columns=target_features)
        elif task_name == 'predict_num_complications':
            
            complications = target_features.copy()
            complications.remove('LET_IS')
            for col in complications:
                data[col] = data[col].astype(float)
            data['num_complications'] = data.apply(lambda row: sum(row[complications]), axis = 1)
            
            target_info = {
                'target': 'num_complications',
                'task_type': 'regression'
            }

            data = data.drop(columns=target_features)
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
            threshold1_num = 0.3,
            threshold2_num = 0.05,
            threshold1_cat = 0.3,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info
    
###################################################################################################################################
# Diabetic Hospital Dataset
###################################################################################################################################
class DiabeticHospitalDataset(Dataset):

    def __init__(self):
        
        name = 'diabetic_hospital'
        subject_area = 'Medical'
        year = 2014
        url = 'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008?'
        download_link = 'https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip'
        description = "The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks"
        notes = 'Diabetic Hospital Dataset'
        data_type = 'mixed'
        self.pub_link = ''
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
        if not os.path.exists(os.path.join(self.data_dir, "diabetic_data.csv")):
            downloader = URLDownloader(url = self.download_link, zipfile = True)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(
            os.path.join(self.data_dir, "diabetic_data.csv"), 
            na_values = ['?',"Unknown/Invalid"], low_memory = False
        )
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = ['age', 'weight']
        ordinal_feature_order_dict = {
            'age': [
                '[90-100)', '[80-90)', '[70-80)', '[60-70)', '[50-60)', '[40-50)', '[30-40)', 
                '[20-30)', '[10-20)', '[0-10)'
            ].reverse(),
            'weight': [
                '>200', '[175-200)', '[150-175)', '[125-150)', '[100-125)', '[75-100)', '[50-75)', 
                '[25-50)', '[0-25)'
            ].reverse()
        }
        binary_features = [
            'gender', 'acetohexamide', 'glipizide-metformin', 'glimepiride-pioglitazone', 
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
        ]
        numerical_features = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
            'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]
        
        multiclass_features = [
            col for col in raw_data.columns 
            if col not in binary_features + numerical_features + ordinal_features
        ]
        
        target_features = ['readmitted']
        sensitive_features = ['race', 'gender', 'age']
        drop_features = ['encounter_id', 'patient_nbr']
        task_names = ['predict_readmission', 'predict_readmission_30']
        
        feature_groups = {
            'demongraphic': [
                'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 
                'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'readmitted'
            ],
            'clinical_info': [
                'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                'number_outpatient', 'number_emergency',  'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 
                'number_diagnoses'
            ],
            'diagnosis': [
                'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 
                'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
            ]
        }
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

        if task_name == 'predict_readmission':
            data['readmitted_binary'] = data['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
            target_info = {
                'target': 'readmitted_binary',
                'task_type': 'classification'
            }
            data = data.drop(columns=['readmitted'])
        elif task_name == 'predict_readmission_30':
            data['readmitted_30_days'] = data['readmitted'].apply(lambda x: 0 if x == '<30' else 1)
            target_info = {
                'target': 'readmitted_30_days',
                'task_type': 'classification'
            }
            data = data.drop(columns=['readmitted'])
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
        data_config = data_config.copy()
        # drop features
        drop_cols = [
            'patient_nbr', 'diag_2', 'diag_3', 'encounter_id',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
            'glipizide-metformin', 'examide', 'citoglipton', 'troglitazone', 'tolazamide',
            'miglitol', 'acarbose', 'tolbutamide', 'acetohexamide', 'chlorpropamide', 
            'nateglinide', 'repaglinide'
        ]
        
        data = data.drop(columns=drop_cols)
        feature_mapping = {}
        
        ################################################################################################
        # Feature Engineering
        ################################################################################################
        # Admission and Discharge
        data = data[data["admission_type_id"].isin(['1', '2', '3', '5', '6'])].copy()
        data.loc[:, "discharge_disposition_id"] = (
            data.discharge_disposition_id.apply(lambda x:'1' if x=='1' else '0')
        )
        data.loc[:, "admission_source_id"] = data["admission_source_id"].apply(
            lambda x: '0' if x in ['1', '2', '3'] else ('1' if x == '7' else '2')
        ).astype(str)

        # Medical Specialty
        specialties = [
            "Missing",
            "InternalMedicine",
            "Emergency/Trauma",
            "Family/GeneralPractice",
            "Cardiology",
            "Surgery"
        ]
        data["medical_specialty"] = data["medical_specialty"].apply(lambda x: x if x in specialties else "Other")
        
        # Diagnosis
        data.loc[:, "diag_1"] = data["diag_1"].replace(
            regex={
                "[7][1-3][0-9]": "Musculoskeltal Issues",
                "250.*": "Diabetes",
                "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
                "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues"
            }
        )
        diagnoses = ["Respitory Issues", "Diabetes", "Genitourinary Issues", "Musculoskeltal Issues"]
        data["diag_1"] = data["diag_1"].apply(lambda x: x if x in diagnoses else "Other")
        
        # Payer Code
        data.loc[:, "medicare"] = data["payer_code"].apply(lambda x: 1 if x == 'MD' else 0).astype(str)
        data.loc[:, "medicaid"] = data["payer_code"].apply(lambda x: 1 if x == 'MC' else 0).astype(str)
        feature_mapping['medicare'] = 'payer_code'
        feature_mapping['medicaid'] = 'payer_code'
        
        # Binning
        data.loc[:, "had_emergency"] = (data["number_emergency"] > 0).astype(str)
        data.loc[:, "had_inpatient_days"] = (data["number_inpatient"] > 0).astype(str)
        data.loc[:, "had_outpatient_days"] = (data["number_outpatient"] > 0).astype(str)
        data = data.drop(columns = ['number_emergency', 'number_inpatient', 'number_outpatient'])
        feature_mapping['had_emergency'] = 'number_emergency'
        feature_mapping['had_inpatient_days'] = 'number_inpatient'
        feature_mapping['had_outpatient_days'] = 'number_outpatient'
        
        data_config['binary_features'].extend(
            ['medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days']
        )
        
        ordinal_as_numerical = ml_task_prep_config.ordinal_as_numerical
        feature_type_handler = BasicFeatureTypeHandler(ordinal_as_numerical)
        
        data, numerical_features, categorical_features = feature_type_handler.handle_feature_type(
            data, data_config
        )
        
        return data, {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'feature_mapping': feature_mapping
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
        missing_data_info = {}
        for col in ['A1Cresult', 'medical_specialty', 'payer_code']:
            if col in data.columns:
                ms_ratio = data[col].isnull().sum() / len(data)
                data[col] = data[col].fillna('Missing')
                missing_data_info[col] = {
                    'action': 'fillna',
                    'missing_ratio': ms_ratio / len(data)
                }
        
        for col in ['weight', 'max_glu_serum']:
            if col in data.columns:
                ms_ratio = data[col].isnull().sum() / len(data)
                data = data.drop(col, axis = 1)
                missing_data_info[col] = {
                    'action': 'drop_col',
                    'missing_ratio': ms_ratio
                }
                
        for col in data.columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > 0:
                missing_data_info[col] = {
                    'action': 'drop_row',
                    'missing_ratio': missing_ratio
                }
                
        return data.dropna(), missing_data_info


###################################################################################################################################
# ARI2 Dataset
###################################################################################################################################
class ARI2Dataset(Dataset):

    def __init__(self):
        
        name = 'ari2'
        subject_area = 'Medical'
        year = 2024
        url = 'https://hbiostat.org/data/'
        download_link = 'https://hbiostat.org/data/repo/ari.zip'
        description = "WHO ARI Multicentre Study of clinical signs and etiologic agents"
        notes = 'Clinical Sign, Etiologic Agent'
        data_type = 'mixed'
        self.pub_link = 'https://journals.lww.com/pidj/Fulltext/1999/10001/Clinical_prediction_of_serious_bacterial.5.aspx'
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
        if not (
            os.path.exists(os.path.join(self.data_dir, 'ari', 'ari.csv')) and
            os.path.exists(os.path.join(self.data_dir, 'ari', 'Y.csv')) and
            os.path.exists(os.path.join(self.data_dir, 'ari', 'Y.death.csv'))
        ):
            downloader = URLDownloader(url = self.download_link)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        feature_file_path = os.path.join(self.data_dir, 'ari', 'ari.csv')
        target_file_path = os.path.join(self.data_dir, 'ari', 'Y.csv')
        target2_file_path = os.path.join(self.data_dir, 'ari', 'Y.death.csv')
        
        try:
            feature_data = pd.read_csv(feature_file_path, index_col=0)
            target_data = pd.read_csv(target_file_path, index_col=0)
            target2_data = pd.read_csv(target2_file_path, index_col=0)
        except Exception as e:
            print(e)
            return False

        feature_data = feature_data.reset_index(drop=True)
        target_data = target_data.reset_index(drop=True)
        target2_data = target2_data.reset_index(drop=True)
        
        raw_data = feature_data
        raw_data['Y'] = target_data
        raw_data['Y_death'] = target2_data
        raw_data = raw_data.drop(columns=['stno', 'dead'])
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        multiclass_features = [
            "weight", "clin", "cdip", "country", "cprot", "hlt", "impcl", "saogp", "hfa", "ldy", "inc", "sr1", "sr2",
            "lcw", "nfl", "str", "gru", "csd", "aro", "qcr", "con", "att", "mvm", "afe", "absu", "stu", "deh", "dcp", "crs",
            "skr", "hyp", "smi2", "abd", "nut", "oto", "ova", "adt", "Y", "Y_death"
        ]
        numerical_features = [
            "wbco", "lpcc", "illd", "biwt", "hcir", "wght", "lgth", "temp", "hrat", "age", "rr", "pmcr", "daydth", "waz", "wam", "bcpc",
            "nxray", "s1", "s2", "s3", "pneu"
        ]
        
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Y', 'Y_death']
        sensitive_features = []
        drop_features = []
        task_names = ['predict_Y', 'predict_Y_death']
        
        feature_groups = {
            'demongraphic': [
                'weight', 'age', 'country', 'wght', 'lgth', 'waz', 'wam', 'biwt', 'hcir', 'bat',
                'smi2', 'mvm', 'afe', 'absu', 'nut', 'Y', 'Y_death'    
            ],
            'past_history': [
                'illd', 'daydth', 'cprot', 'cdip', 'clin', 'impcl', 'saogp', 'omph', 'conj', 
                'hfa', 'hfb', 'hfe', 'hap', 'hcl', 'hcm', 'hcs', 'hdi', 'hvo', 'hbr', 'fde', 
                'chi', 'twb', 'ldy'
            ], 
            'lab_vital_signs': [
                'lpcc', 'wbco', 'bcpc', 'pmcr', 'temp', 'hrat', 'rr', 'pmcr', 'nxray', 'pneu', 
                's1', 's2', 's3', 'sickj', 'sickl', 'sickl1', 'sickl2', 'sickl3', 'sickc'
            ],
            'clinical_status': [
                'slpm', 'slpl', 'wake', 'convul', 'inc', 'sr1', 'sr2', 'apn', 'lcw', 'nfl', 
                'str', 'gru', 'coh', 'ccy', 'jau', 'csd', 'csa', 'aro', 'qcr', 'con', 'att', 
                'stu', 'deh', 'dcp', 'crs', 'skr', 'abb', 'abk', 'hyp', 'hlt', 'abd', 
                'whz', 'hdb', 'puskin', 'oto', 'ova', 'oab', 'lp.pos', 'adt',
            ],
        }
        fed_cols = ['country']
        
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

        if task_name == 'predict_Y':
            target_info = {
                'target': 'Y',
                'task_type': 'classification'
            }
            data = data.drop(columns=['Y_death', 'saogp'])
        elif task_name == 'predict_Y_death':
            target_info = {
                'target': 'Y_death',
                'task_type': 'classification'
            }
            data = data.drop(columns=['Y', 'saogp'])
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
        
        # drop_cols = ['s1', 's2', 's3', 'nxray', 'pneu', 'pmcr', 'cdip', 'lpcc', 'wbco', 'weight', 'daydth']
        # drop_rows = ['wght', 'cprot', 'impcl', 'saogp', 'bcpc', 'clin', 'bcpc']
        # data = data.drop(columns=drop_cols)
        # data = data.dropna(subset=drop_rows)
        
        # missing_data_info = {}
        # for col in drop_cols:
        #     missing_data_info[col] = {
        #         'action': 'drop_col',
        #         'missing_ratio': data[col].isnull().sum() / len(data)
        #     }
        # for row in drop_rows:
        #     missing_data_info[row] = {
        #         'action': 'drop_row',
        #         'missing_ratio': data.loc[row].isnull().sum() / len(data)
        #     }
            
        # data = data.drop(columns=drop_cols)
        # data = data.dropna(subset=drop_rows)
        
        return data, missing_data_info
    
###################################################################################################################################
# RHC Dataset
###################################################################################################################################
class RHCDataset(Dataset):

    def __init__(self):
        
        name = 'rhc'
        subject_area = 'Medical'
        year = 2024
        url = 'https://hbiostat.org/data/'
        download_link = 'https://hbiostat.org/data/repo/rhc.csv'
        description = "WHO ARI Multicentre Study of clinical signs and etiologic agents"
        notes = 'Clinical Sign, Etiologic Agent'
        data_type = 'mixed'
        self.pub_link = 'https://journals.lww.com/pidj/Fulltext/1999/10001/Clinical_prediction_of_serious_bacterial.5.aspx'
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
        if not os.path.exists(os.path.join(self.data_dir, 'rhc.csv')):
            downloader = URLDownloader(url = self.download_link, zipfile = False)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'rhc.csv'), index_col=0)
        
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
            'death', 'cardiohx', 'chfhx', 'dementhx', 'psychhx', 'chrpulhx', 'renalhx', 'liverhx', 'gibledhx', 'malighx',
            'immunhx', 'transhx', 'amihx', 'sex', 'dth30', 'swang1', 'dnr1', 'resp', 'card', 'neuro', 'gastr', 
            'renal', 'meta', 'hema', 'seps', 'trauma', 'ortho',
        ]
        multiclass_features = [
            'cat1', 'cat2', 'ca', 'race', 'ninsclas', 'ptid', 'income'
        ]
        
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['death', 'dth30']
        sensitive_features = ['age', 'sex', 'race']
        drop_features = ['ptid']
        task_names = ['predict_death', 'predict_dth30']
        
        feature_groups = {
            'demographic': [
                'sadmdte', 'dschdte', 'dthdte', 'lstctdte', 'age', 'sex', 'edu', 'race', 'income', 'ninsclas', 'ptid',
                'death', 'dth30'
            ],
            'admission_diagnosis': [
                'cat1', 'cat2', 'resp', 'card', 'neuro', 'gastr', 'renal', 'meta', 'hema', 'seps', 
                'trauma', 'ortho', 'ca'
            ],
            'comorbidity': [
                'cardiohx', 'chfhx', 'dementhx', 'psychhx', 'chrpulhx', 'renalhx', 'liverhx', 
                'gibledhx', 'malighx', 'immunhx', 'transhx', 'amihx', 'aps1'
            ],
            'clinical_status': [
                'surv2md1', 'das2d3pc', 't3d30', 'scoma1', 'wblc1', 'hrt1', 'resp1', 'meanbp1', 
                'temp1', 'pafi1', 'alb1', 'hema1', 'bili1', 'crea1', 'bili1', 'sod1', 'pot1', 
                'paco21', 'ph1', 'swang1', 'wtkilo1', 'dnr1', 'urin1', 'adld3p'
            ],
        }
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

        if task_name == 'predict_death':
            target_info = {
                'target': 'death',
                'task_type': 'classification'
            }
            data = data.drop(columns=['dth30', 'dthdte', 't3d30'])
        elif task_name == 'predict_dth30':
            target_info = {
                'target': 'dth30',
                'task_type': 'classification'
            }
            data = data.drop(columns=['death', 'dthdte', 't3d30'])
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
        
        # drop features
        drop_cols = ['sadmdte', 'dschdte', 'lstctdte', 'ptid']
        data = data.drop(columns=drop_cols)
        
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
            threshold1_cat = 0.7,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info
    
###################################################################################################################################
# Crash Dataset
###################################################################################################################################
class CrashDataset(Dataset):

    def __init__(self):
        
        name = 'crash'
        subject_area = 'Medical'
        year = 2020
        url = 'https://hbiostat.org/data/'
        download_link = 'https://hbiostat.org/data/repo/crash2.rda'
        description = "This publicly available clinical trial dataset of 20,207 patients was generously supplied by" \
                       "the freeBIRD Bank of Injury and Emergency Research Data from the UK. The CRASH-2 trandomized trial" \
                      "studied antifibrinolytic treatment in significant hemorrhage post trauma."
        notes = 'Crash'
        data_type = 'mixed'
        self.pub_link = ''
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
        if not os.path.exists(os.path.join(self.data_dir, 'crash2.rda')):
            # download data
            downloader = URLDownloader(url = self.download_link, zipfile = False)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        converted = rdata.read_rda(os.path.join(self.data_dir, 'crash2.rda'))
        raw_data = converted['crash2']
        raw_data = raw_data.drop(columns=['entryid'])
        raw_data.columns = [str(col) for col in raw_data.columns]
        raw_data['death'] = pd.isna(raw_data['condition'])
        raw_data['condition'] = raw_data['condition'].fillna(-1)
        raw_data['cause'] = raw_data['cause'].fillna(-1)
        raw_data.replace({pd.NA: np.nan}, inplace=True)
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        binary_features = ["sex", 'death']
        multiclass_features = ['source', "injurytype", 'cause', "condition", 'outcomeid', 'scauseother', 'status']
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['condition', 'death', 'cause']
        sensitive_features = ["age", "sex"]
        drop_features = ['outcomeid', "scauseother", "status", 'ddeath', 'boxid', 'packnum']
        task_names = ['predict_condition', 'predict_death', 'predict_cause']
        
        feature_groups = {
            'demongraphic': [
                'source', 'trandomised', 'outcomeid', 'sex', 'age', 'injurytime', 'injurytype', 'ddeath', 'scauseother',
                'status', 'ddischarge', 'ndaysicu', 'boxid', 'packnum', 'condition', 'cause', 'death'
            ],
            'clinical_status': [
                'sbp', 'rr', 'cc', 'hr',  'gcseye', 'gcsmotor', 'gcsverbal', 'gcs', 
                'bheadinj', 'bneuro', 'bchest', 'babdomen', 'bpelvis', 'bpe', 'bdvt', 'bstroke', 
                'bbleed', 'bmi', 'bgi', 'bloading', 'bmaint', 'btransf', 'ncell', 'nplasma', 'nplatelets', 
                'ncryo', 'bvii'
            ],
        }
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

        if task_name == 'predict_death':
            target_info = {
                'target': 'death',
                'task_type': 'classification'
            }
            data = data.drop(columns=['condition', 'cause', 'ddischarge'])
        elif task_name == 'predict_condition':
            target_info = {
                'target': 'condition',
                'task_type': 'classification'
            }
            data = data.drop(columns=['death', 'ddischarge', 'cause'])
        elif task_name == 'predict_cause':
            target_info = {
                'target': 'cause',
                'task_type': 'classification'
            }
            data = data.drop(columns=['death', 'condition', 'ddischarge'])
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
        # drop features
        data = data.drop(['outcomeid', "scauseother", "status", 'ddeath', 'boxid', 'packnum'], axis = 1)
        
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
# Support
###################################################################################################################################
class SupportDataset(Dataset):

    def __init__(self):
        
        name = 'support'
        subject_area = 'Medical'
        year = 2020
        url = 'https://hbiostat.org/data/'
        download_link = 'https://hbiostat.org/data/repo/support2csv.zip'
        description = "Support dataset"
        notes = 'Support'
        data_type = 'numerical'
        self.pub_link = 'The support dataset is a random sample of 9000 patients from Phases I & II of SUPPORT' \
                        '(Study to Understand Prognoses Preferences Outcomes and Risks of Treatment).'
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
        if not os.path.exists(os.path.join(self.data_dir, 'support2.csv')):
            downloader = URLDownloader(url = self.download_link, zipfile = True)
            download_status = downloader._custom_download(data_dir = self.data_dir)
            if not download_status:
                raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'support2.csv')).reset_index(drop=True)
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = []
        ordinal_feature_order_dict = {}
        binary_features = ['sex', 'hospdead', 'diabetes', 'dementia']
        multiclass_features = ['dzgroup', 'dzclass','race', 'ca', 'dnr', 'sfdm2', 'income', 'death']
        
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['hospdead', 'hday', 'death']
        sensitive_features = ['age', 'sex', 'race']
        drop_features = []
        task_names = ['predict_death', 'predict_hospdead', 'predict_hday']
        
        feature_groups = {
            'demographic': [
                'age', 'sex', 'race', 'edu', 'income', 'slos', 'd.time', 'charges', 'totcst', 'totmcst',
                'dnr', 'dnrday', 'death', 'hospdead', 'hday'
            ],
            'comorbidity': ['dzgroup', 'dzclass', 'num.co', 'scoma', 'diabetes', 'dementia', 'ca'],
            'clinical_status': [
                'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili',
                'avtisst', 'sps', 'aps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls', 'sfdm2', 'adlsc'
            ],
        }
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
        drop_cols = ['sps', 'aps', 'prg2m', 'prg6m', 'surv2m', 'surv6m', 'd.time', 'dnr', 'dnrday']
        
        if task_name == 'predict_hospdead':
            target_info = {
                'target': 'hospdead',
                'task_type': 'classification'
            }
            data = data.drop(columns=['hday', 'death', 'sfdm2'] + drop_cols)
        elif task_name == 'predict_hday':
            target_info = {
                'target': 'hday',
                'task_type': 'regression'
            }
            data = data.drop(columns=['hospdead', 'death'] + drop_cols)
        elif task_name == 'predict_death':
            target_info = {
                'target': 'death',
                'task_type': 'classification'
            }
            data = data.drop(columns=['hospdead', 'hday'] + drop_cols)
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
            threshold1_num = 0.6,
            threshold2_num = 0.05,
            threshold1_cat = 0.4,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info
    
###################################################################################################################################
# CIBMTR HCT Survival Dataset
###################################################################################################################################
class CIBMTRHCTSurvivalDataset(Dataset):

    def __init__(self):
        
        name = 'cibmtr_hct_survival'
        subject_area = 'Medical'
        year = 2017
        url = 'https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/data'
        download_link = None
        description = "The dataset consists of 59 variables related to hematopoietic stem cell transplantation (HSCT)" \
                      "encompassing a range of demographic and medical characteristics of both recipients and donors" \
                      ", such as age, sex, ethnicity, disease status, and treatment details. "
        notes = 'HSCT, Survival Prediction'
        data_type = 'mixed'
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
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset and specify meta data information
        
        Returns:
            raw_data: pd.DataFrame, raw data
            meta_data: dict, meta data
        """
        # download data
        downloader = KaggleDownloader(
            dataset_name = 'equity-post-HCT-survival-predictions',
            file_names = ['train.csv', 'data_dictionary.csv'],
            download_all = True,
            competition = True
        )
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(
            self.data_dir, 'train.csv'), index_col = 0, na_values = ['nan', np.nan]
        )
        raw_data = raw_data.reset_index(drop = True)
        raw_data = raw_data.replace('nan', np.nan)
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        data_dictionary = pd.read_csv(os.path.join(self.data_dir, 'data_dictionary.csv'))
        
        numerical_features = [
            'hla_high_res_8', 'hla_high_res_6', 'hla_low_res_8', 'hla_low_res_6', 'hla_high_res_10', 'hla_low_res_10',
            'year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score', 
            'efs_time'
        ]
        
        binary_features = []
        multiclass_features = []
        for column in raw_data.columns:
            if column in numerical_features:
                continue
            else:
                if raw_data[column].nunique() == 2:
                    binary_features.append(column)
                elif raw_data[column].nunique() > 2:
                    multiclass_features.append(column)
                    
        ordinal_features = []
        ordinal_feature_order_dict = {}
        
        target_features = ['efs', 'efs_time']
        
        sensitive_features = ['ethnicity', 'race_group']
        drop_features = []
        task_names = ['predict_survival', 'predict_efs_time']
        
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
            data['efs'] = data['efs'].astype(float)
            fitter.fit(data['efs_time'], data['efs'])
            target = fitter.predict(data['efs_time']).reset_index(drop = True)
            data['survival_risk'] = target
            target_info = {
                'target': 'survival_risk',
                'task_type': 'regression'
            }
            data = data.drop(columns = ['efs', 'efs_time'])
        elif task_name == 'predict_efs_time':
            target_info = {
                'target': 'efs_time',
                'task_type': 'regression'
            }
            data = data.drop(columns = ['efs'])
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
            threshold1_num = 0.6, threshold2_num = 0.05, 
            threshold1_cat = 0.8, threshold2_cat = 0.05
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        assert data.isna().sum().sum() == 0, "Missing data is not handled"
        
        return data, {}

###################################################################################################
# HCC Survival Dataset
###################################################################################################
class HCCSurvivalDataset(Dataset):

    def __init__(self):
        
        name = 'hcc_survival'
        subject_area = 'Medical'
        year = 2017
        url = 'https://archive.ics.uci.edu/dataset/423/hcc+survival'
        download_link = 'https://archive.ics.uci.edu/static/public/423/hcc+survival.zip'
        description = "Hepatocellular Carcinoma dataset (HCC dataset) was collected at a University Hospital in Portugal." \
                      "It contains real clinical data of 165 patients diagnosed with HCC."
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
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'hcc-survival', 'hcc-data.txt'), header=None, na_values='?')
        
        column_names = [
            "Gender", "Symptoms", "Alcohol",
            "Hepatitis B Surface Antigen", "Hepatitis B e Antigen", "Hepatitis B Core Antibody", "Hepatitis C Virus Antibody",
            "Cirrhosis", "Endemic Countries", "Smoking", "Diabetes", "Obesity", "Hemochromatosis", "Arterial Hypertension",
            "Chronic Renal Insufficiency", "Human Immunodeficiency Virus", "Nonalcoholic Steatohepatitis", "Esophageal Varices",
            "Splenomegaly", "Portal Hypertension", "Portal Vein Thrombosis", "Liver Metastasis", "Radiological Hallmark",
            "Age", "Grams of Alcohol per day", "Packs of cigarets per year", "Performance Status","Encefalopathy degree",
            "Ascites degree", "International Normalised Ratio", "Alpha-Fetoprotein(ng/mL)", "Haemoglobin(g/dL)",
            "Mean Corpuscular Volume(fl)", "Leukocytes(G/L)", "Platelets(G/L)", "Albumin (mg/dL)", "Total Bilirubin(mg/dL)",
            "Alanine transaminase(U/L)", "Aspartate transaminase(U/L)", "Gamma glutamyl transferase(U/L)",
            "Alkaline phosphatase(U/L)", "Total Proteins(g/dL)", "Creatinine(mg/dL)", "Number of Nodules",
            "Major dimension of nodule(cm)", "Direct Bilirubin(mg/dL)", "Iron(mcg/dL)", "Oxygen Saturation(%)",
            "Ferritin(ng/mL)", "Class"
        ]
        raw_data.columns = column_names
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
        """
        Set raw data configuration
        
        Returns:
            raw_data_config: dict, raw data configuration
        """
        ordinal_features = ['Performance Status', 'Encefalopathy degree', 'Ascites degree']
        ordinal_feature_order_dict = {
            'Performance Status': [0, 1, 2, 3, 4],
            'Encefalopathy degree': [1, 2, 3],
            'Ascites degree': [1, 2, 3]
        }
        multiclass_features = []
        binary_features = ['Gender', 'Class', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 
            'Hepatitis C Virus Antibody', 'Hepatitis B Core Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 
            'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 
            'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 
            'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark']
        
        numerical_features = [
            col for col in raw_data.columns 
            if col not in binary_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['Class']
        sensitive_features = ['Gender', 'Age']
        drop_features = []
        task_names = ['predict_class']
        
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

        if task_name == 'predict_class':
            target_info = {
                'target': 'Class',
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
            threshold1_num = 0.6,
            threshold2_num = 0.05,
            threshold1_cat = 0.7,
            threshold2_cat = 0.05,
            impute_num = 'mean',
            impute_cat = 'other'
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        
        return data, missing_data_info
    
###################################################################################################################################
# ZAlizadehsani Dataset
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
        task_names = ['LAD', 'LCX', 'RCA', 'Cath']
        
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
# NasarianCAD Dataset
###################################################################################################################################
class NasarianCADDataset(Dataset):

    def __init__(self):
        
        name = 'nasariancad'
        subject_area = 'Medical'
        year = 2022
        url = 'https://www.kaggle.com/datasets/elhamnasarian/nasarian-cad-dataset'
        download_link = None
        description = "First Heart Disease Dataset with Occupational Risk Factors + Clinical Features"
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
        
    def _load_raw_data(self):
        
        # download data
        downloader = KaggleDownloader(
            dataset_name = 'elhamnasarian/nasarian-cad-dataset',
            file_names = ['nasariancad.csv'],
            download_all = True
        )
        download_status = downloader._custom_download(data_dir = self.data_dir)
        if not download_status:
            raise Exception(f'Failed to download data for {self.name}')
        
        # load raw data
        raw_data = pd.read_csv(os.path.join(self.data_dir, 'nasariancad.csv'))
        raw_data.rename(columns = {'Function\n': 'Function'}, inplace = True)
        
        return raw_data
    
    def _set_raw_data_config(self, raw_data: pd.DataFrame) -> dict:
    
        # Specify meta data
        numerical_features = [
            'Age', 'Weight', 'Length', 'BMI', 'BP', 'PR', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'RBC', 'HB', 'POLY',
            'WBC', 'PLT', 'HTC', 'Lymph', 'FBS'
        ]
        multiclass_features = ['eo', 'Lungrales']
        ordinal_features = []
        ordinal_feature_order_dict = {}
        binary_features = [
            col for col in raw_data.columns 
            if col not in numerical_features + multiclass_features + ordinal_features
        ]
        
        target_features = ['angiographyCAD', 'heartattack']
        sensitive_features = ['Age']
        drop_features = []
        task_names = ['predict_cad', 'predict_heartattack']
        
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
        if task_name == 'predict_cad':
            
            target_info = {
                'target': 'angiographyCAD',
                'task_type': 'classification'
            }
        elif task_name == 'predict_heartattack':
            target_info = {
                'target': 'heartattack',
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
        missing_data_handler = BasicMissingDataHandler(
            threshold1_num = 0.5, 
            threshold2_num = 0.2, 
            threshold1_cat = 0.5, 
            threshold2_cat = 0.2
        )
        data, missing_data_info = missing_data_handler.handle_missing_data(data, categorical_features)
        assert data.isna().sum().sum() == 0, "Missing data is not handled"
        return data, missing_data_info