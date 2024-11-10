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
