import pytest
import pandas as pd

from MedDataKit.dataset.clinical_dataset import (
    ZigongHeartFailureDataset,
    MIMIC2IACDataset,
    MIComplicationsDataset,
    DiabeticHospitalDataset,
    ARI2Dataset,
    SupportDataset,
    CIBMTRHCTSurvivalDataset,
    CrashDataset,
    RHCDataset,
    HCCSurvivalDataset,
    ZAlizadehsaniDataset,
    NasarianCADDataset
)

from MedDataKit.dataset.repo_dataset import (
    ArrhythmiaDataset,
    ColposcopyDataset,
    SPECTFDataset,
    BreastCancerWisconsinDataset,
    DermatologyDataset,
    BoneTransplantDataset,
    ParkinsonsDataset,
    CervicalRiskDataset,
    BacteremiaDataset,
    FetalCTGDataset
)

from MedDataKit.dataset.bioinfo_dataset import (
    CodonUsageDataset,
    GENE3494Dataset
)

@pytest.mark.parametrize("dataset_class", [
    # --------------------------------------------------------
    # clinical dataset
    # --------------------------------------------------------
    ZigongHeartFailureDataset,
    MIMIC2IACDataset,
    MIComplicationsDataset,
    DiabeticHospitalDataset,
    ARI2Dataset,
    SupportDataset,
    CIBMTRHCTSurvivalDataset,
    CrashDataset,
    RHCDataset,
    HCCSurvivalDataset,
    ZAlizadehsaniDataset,
    NasarianCADDataset,
    # --------------------------------------------------------
    # repo dataset
    # --------------------------------------------------------
    ArrhythmiaDataset,
    ColposcopyDataset,
    SPECTFDataset,
    BreastCancerWisconsinDataset,
    DermatologyDataset,
    BoneTransplantDataset,
    ParkinsonsDataset,
    CervicalRiskDataset,
    BacteremiaDataset,
    FetalCTGDataset,
    # --------------------------------------------------------
    # bioinfo dataset
    # --------------------------------------------------------
    CodonUsageDataset,
    GENE3494Dataset
])
def test_load_raw_data(dataset_class):
    dataset = dataset_class()
    data = dataset.load_raw_data()
    print(data.shape)
    assert data is not None
    assert len(data) > 0
    assert isinstance(data, pd.DataFrame)



@pytest.mark.parametrize("dataset_class", [
    # --------------------------------------------------------
    # clinical dataset
    # --------------------------------------------------------
    ZigongHeartFailureDataset,
    MIMIC2IACDataset,
    MIComplicationsDataset,
    DiabeticHospitalDataset,
    ARI2Dataset,
    SupportDataset,
    CIBMTRHCTSurvivalDataset,
    CrashDataset,
    RHCDataset,
    HCCSurvivalDataset,
    ZAlizadehsaniDataset,
    NasarianCADDataset,
    # --------------------------------------------------------
    # repo dataset
    # --------------------------------------------------------
    ArrhythmiaDataset,
    ColposcopyDataset,
    SPECTFDataset,
    BreastCancerWisconsinDataset,
    DermatologyDataset,
    BoneTransplantDataset,
    ParkinsonsDataset,
    CervicalRiskDataset,
    BacteremiaDataset,
    FetalCTGDataset,
    # --------------------------------------------------------
    # bioinfo dataset
    # --------------------------------------------------------
    CodonUsageDataset,
    GENE3494Dataset
])

def test_load_ml_task_data(dataset_class):
    dataset = dataset_class()
    dataset.load_raw_data()
    task_names = dataset.get_task_names()
    for task_name in task_names:
        dataset.generate_ml_task_dataset(
            task_name = task_name, 
            config = {
                'numerical_encoding': 'quantile', 
                'missing_strategy': 'impute', 
                'ordinal_as_numerical': False,
                'categorical_encoding': 'ordinal'
            }, 
            verbose = True
        )
        data = dataset.ml_task_dataset.data
        config = dataset.ml_task_dataset.data_config
        
        assert data is not None
        assert config is not None
        assert len(data) > 0
        assert isinstance(data, pd.DataFrame)
        assert isinstance(config, dict)
        assert pd.isna(data).sum().sum() == 0
        

@pytest.mark.parametrize("dataset_class", [
    # --------------------------------------------------------
    # clinical dataset
    # --------------------------------------------------------
    ZigongHeartFailureDataset,
    MIMIC2IACDataset,
    MIComplicationsDataset,
    DiabeticHospitalDataset,
    ARI2Dataset,
    SupportDataset,
    CIBMTRHCTSurvivalDataset,
    CrashDataset,
    RHCDataset,
    HCCSurvivalDataset,
    ZAlizadehsaniDataset,
    NasarianCADDataset,
    # --------------------------------------------------------
    # repo dataset
    # --------------------------------------------------------
    ArrhythmiaDataset,
    ColposcopyDataset,
    SPECTFDataset,
    BreastCancerWisconsinDataset,
    DermatologyDataset,
    BoneTransplantDataset,
    ParkinsonsDataset,
    CervicalRiskDataset,
    BacteremiaDataset,
    FetalCTGDataset,
    # --------------------------------------------------------
    # bioinfo dataset
    # --------------------------------------------------------
    CodonUsageDataset,
    GENE3494Dataset
])
def test_load_ml_task_data_onehot(dataset_class):
    dataset = dataset_class()
    dataset.load_raw_data()
    task_names = dataset.get_task_names()
    for task_name in task_names:
        dataset.generate_ml_task_dataset(
            task_name = task_name, 
            config = {
                'numerical_encoding': 'quantile', 
                'missing_strategy': 'impute', 
                'ordinal_as_numerical': False,
                'categorical_encoding': 'onehot'
            }, 
            verbose = True
        )
        data = dataset.ml_task_dataset.data
        config = dataset.ml_task_dataset.data_config
        
        assert data is not None
        assert config is not None
        assert len(data) > 0
        assert isinstance(data, pd.DataFrame)
        assert isinstance(config, dict)
        assert pd.isna(data).sum().sum() == 0
        for col in config['categorical_columns']:
            assert len(data[col].unique()) == 2, f"Categorical column {col} should have exactly 2 classes"
        






