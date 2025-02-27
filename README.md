# MediDataKit

## Design

1. Downloader: download data from specific sources e.g. UCIML, Local Directory
    - Functionalities:
        - download data files with provided URLs, Paths, and other information
        - save downloaded files, check if the file exists before downloading
    - Information produced:
        - raw_data_files: original downloaded data files (file name and paths, number of files, file size, etc.)

2. DataLoader: load data into memory
    - Functionalities:
        - load downloaded data into memory


raw_data_files (original downloaded data files) -> raw_data (centralized data, federated data) -> ML-ready data (data ready for performing ML tasks)


## Included Datasets

### clinical dataset

- ZigongHeartFailureDataset
- MIMIC2IACDataset
- MIComplicationsDataset
- DiabeticHospitalDataset
- ARI2Dataset
- SupportDataset
- CIBMTRHCTSurvivalDataset
- CrashDataset
- RHCDataset
- HCCSurvivalDataset
- ZAlizadehsaniDataset
- NasarianCADDataset

### repo dataset

- ArrhythmiaDataset
- ColposcopyDataset
- SPECTFDataset
- BreastCancerWisconsinDataset
- DermatologyDataset
- BoneTransplantDataset
- ParkinsonsDataset
- CervicalRiskDataset
- BacteremiaDataset
- FetalCTGDataset

### bioinfo dataset

- CodonUsageDataset
- GENE3494Dataset

## APP

```
shiny run --reload --launch-browser shiny-app/app.py
```