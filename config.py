import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
print(DATA_DIR)
DATA_DOWNLOAD_DIR = 'downloaded_data'