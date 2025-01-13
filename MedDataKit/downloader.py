from sklearn.datasets import fetch_openml
import os
import pandas as pd
from abc import ABC, abstractmethod

import tempfile
import tarfile
import zipfile
import os
import pandas as pd
import requests
import io
import pyreadr
import shutil
import requests
import zipfile
import io
import os
import kaggle as kg
from dotenv import load_dotenv

from config import DATA_DIR, DATA_DOWNLOAD_DIR


class DownLoader(ABC):

    def __init__(self):
        pass

    def download(self, data_save_dir: str):

        # check if the data file exists
        if os.path.exists(data_save_dir):
            # if the file exists, load and return the data frame
            data = pd.read_csv(data_save_dir)
            return data
        else:
            # if the file does not exist, call the custom download function
            data = self._custom_download()
            
            # ensure the target directory exists
            os.makedirs(os.path.dirname(data_save_dir), exist_ok=True)
            
            # save the data to the specified path
            data.to_csv(os.path.join(data_save_dir, 'raw_data.csv'), index=False)
            
            return data

    @abstractmethod
    def _custom_download(self):
        """
        Customized download function to download the data from different sources
        """
        pass


class UCIMLDownloader(DownLoader):

    def __init__(self, url: str):
        self.url = url
        super().__init__()

    def _custom_download(self, data_dir: str):

        try:
            download_dir = os.path.join(data_dir)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            url = self.url
            zipfile_name = url.split('/')[-1]
            zipfile_path = os.path.join(download_dir, zipfile_name)
            # check if the zip file exists, if not, download and save zip file
            if not os.path.exists(zipfile_path):
                # download data and unzip to download_dir
                response = requests.get(url)
                
                # save response content to zipfile_path
                with open(zipfile_path, 'wb') as f:
                    f.write(response.content)

            # unzip the zip file
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
                
            # return True if the download is successful
            return True
        except Exception as e:
            print(e)
            return False


class KaggleDownloader(DownLoader):
    
    """
    Downloader for Kaggle datasets
    
    Args:
        dataset_name: str, name of the dataset - e.g 'elhamnasarian/nasarian-cad-dataset
        file_names: list[str], names of the files to download
        download_all: bool, whether to download all files
    
    """

    def __init__(
        self, dataset_name: str, 
        file_names: list[str], 
        download_all: bool = False,
        competition: bool = False
    ):
        
        self.dataset_name = dataset_name
        self.file_names = file_names
        self.download_all = download_all
        self.competition = competition
        super().__init__()
        
    def _custom_download(self, data_dir: str):

        try:
            # load kaggle username and key from .env file
            dotenv_path = '.env'
            load_dotenv(dotenv_path)
            kaggle_username = os.getenv('KAGGLE_USERNAME')
            kaggle_key = os.getenv('KAGGLE_KEY')

            # init kaggle api by setting kaggle.json
            KaggleDownloader.init_on_kaggle(kaggle_username, kaggle_key)
            kg.api.authenticate()

            # download data based on the parameters    
            download_dir = os.path.join(data_dir)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            if not self.competition:
                dataset_name = self.dataset_name
                file_names = self.file_names
                download_all = self.download_all

                # download all files as whole
                if download_all:
                    zipfile_name = dataset_name.split('/')[-1] + '.zip'
                    zipfile_path = os.path.join(download_dir, zipfile_name)
                    if not os.path.exists(zipfile_name):
                        kg.api.dataset_download_files(dataset_name, path=download_dir, unzip = False)
                    
                    # unzip zip file
                    zip_file_path = os.path.join(download_dir, zipfile_name)
                    print(zip_file_path)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(download_dir)
                # download only specific files by file_names
                else:
                    for file_name in file_names:
                        if not os.path.exists(os.path.join(download_dir, file_name)):
                            kg.api.dataset_download_file(dataset_name, file_name, path=download_dir)
            else:
                competition_name = self.dataset_name
                file_names = self.file_names
                download_all = self.download_all

                if download_all:
                    zipfile_name = competition_name.split('/')[-1] + '.zip'
                    zipfile_path = os.path.join(download_dir, zipfile_name)
                    if not os.path.exists(zipfile_name):
                        kg.api.competition_download_files(competition_name, path=download_dir, quiet = False)
                    
                    # unzip zip file
                    zip_file_path = os.path.join(download_dir, zipfile_name)
                    print(zip_file_path)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(download_dir)
                else:
                    for file_name in file_names:
                        if not os.path.exists(os.path.join(download_dir, file_name)):
                            kg.api.competition_download_file(competition_name, file_name, path=download_dir)
                        
            return True
        except Exception as e:
            print(e)
            return False
        
    
    @staticmethod
    def init_on_kaggle(username, api_key):

        import os
        import json
        import subprocess

        # set kaggle config dir based on os
        if os.name == 'nt':
            # windows
            KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('%USERPROFILE%'), '.kaggle')
        else:
            KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
        
        os.makedirs(KAGGLE_CONFIG_DIR, exist_ok = True)
        api_dict = {"username":username, "key":api_key}
        file_path = os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json')
        print(file_path)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(api_dict, f)
        
        # code for permission setting
        # if os.name == 'nt':
        #     # windows -> file path is different
        #     cmd = f"chmod 600 {file_path}"
        # else:
        #     cmd = f"chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json"
        # print(cmd)
        # output = subprocess.check_output(cmd.split(" "))
        # output = output.decode(encoding='UTF-8')
        # print(output)


class RDataDownloader(DownLoader):

    def __init__(self, package_url: str, dataset_path_in_package: str):
        self.package_url = package_url
        self.dataset_path_in_package = dataset_path_in_package
        super().__init__()

    def _custom_download(self, data_dir: str):
        
        download_dir = os.path.join(data_dir, DATA_DOWNLOAD_DIR)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        dataset_path = self.dataset_path_in_package.split('/')[-1]

        # if dataset_path exists, return True
        if os.path.exists(os.path.join(download_dir, dataset_path)):
            return True
        # if dataset_path does not exist, download the package and extract the dataset
        else:
            # create a temporary directory
            temp_dir = tempfile.mkdtemp()
            error_happened = False
            try:
                # download the package
                response = requests.get(self.package_url)
                tar_content = io.BytesIO(response.content)
                
                # extract the package
                with tarfile.open(fileobj=tar_content, mode='r:gz') as tar:
                    tar.extractall(path=temp_dir)
                
                # read the .rda file
                rda_path = os.path.join(temp_dir, self.dataset_path)

                # copy the .rda file to download_dir
                shutil.copy(rda_path, download_dir)
                
            except Exception as e:
                print(e)
                error_happened = True
            finally:
                # delete the temporary directory
                shutil.rmtree(temp_dir)
                return error_happened


class OpenMLDownloader(DownLoader):

    def __init__(self, data_id: int):
        self.data_id = data_id
        super().__init__()

    def _custom_download(self, data_dir: str):

        try:
            download_dir = os.path.join(data_dir, DATA_DOWNLOAD_DIR)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            X, y = fetch_openml(data_id=self.data_id, as_frame=True, return_X_y=True)
            data = X.join(y)

            # save the data to the specified path
            data.to_csv(os.path.join(download_dir, 'data.csv'), index=False)

            return True
        
        except Exception as e:
            print(e)
            return False
        
        
class LocalDownloader(DownLoader):

    def __init__(self, local_data_dir: str, file_names: list[str]):
        self.local_data_dir = local_data_dir
        self.file_names = file_names
        super().__init__()

    def _custom_download(self, data_dir: str):
        pass
