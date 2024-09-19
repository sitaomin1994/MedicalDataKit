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

    def __init__(self, url: str, file_name: str, header: bool = True):
        self.url = url
        self.file_name = file_name
        self.header = header
        super().__init__()

    def _custom_download(self):
        import requests
        import zipfile
        import tempfile
        import io
        import os
        import shutil
        url = self.url
        response = requests.get(url)
        zip_content = io.BytesIO(response.content)  
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        file_name = self.file_name
        file_path = os.path.join(temp_dir, file_name)
        if file_name.endswith('.data'):
            if self.header:
                data = pd.read_csv(file_path)
            else:
                data = pd.read_csv(file_path, header=None)
        
        # remove temp_dir
        shutil.rmtree(temp_dir)

        return data
        


class KaggleDownloader(DownLoader):

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        super().__init__()
        
    def _custom_download(self):
        pass


class RDataDownloader(DownLoader):

    def __init__(self, package_url: str, dataset_path: str):
        self.package_url = package_url
        self.dataset_path = dataset_path
        super().__init__()

    def _custom_download(self):
        
        # create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # download the package
            response = requests.get(self.package_url)
            tar_content = io.BytesIO(response.content)
            
            # extract the package
            with tarfile.open(fileobj=tar_content, mode='r:gz') as tar:
                tar.extractall(path=temp_dir)
            
            # read the .rda file
            rda_path = os.path.join(temp_dir, self.dataset_path)
            
            # use pyreadr to read the .rda file
            result = pyreadr.read_r(rda_path)
            
            # pyreadr returns a dictionary, we take the value of the first key as the DataFrame
            self.data = next(iter(result.values()))
        
        finally:
            # delete the temporary directory
            shutil.rmtree(temp_dir)
        
        self.data.reset_index(drop=True, inplace=True)
        
        return self.data


class OpenMLDownloader(DownLoader):

    def __init__(self, data_id: int):
        self.data_id = data_id
        self.data = None
        super().__init__()

    def _custom_download(self):

        X, y = fetch_openml(data_id=self.data_id, as_frame=True, return_X_y=True)
        self.data = X.join(y)
        return self.data
