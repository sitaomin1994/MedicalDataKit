from sklearn.datasets import fetch_openml

class UCIMLDownloader:

    def __init__(self, url: str):
        self.url = url

    def download(self):
        pass

    def load(self):
        pass


class KaggleDownloader:

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    def download(self):
        pass

    def load(self):
        pass


class RDataDownloader:

    def __init__(self, package_url: str, dataset_path: str):
        self.package_url = package_url
        self.dataset_path = dataset_path

    def download(self):
        import tempfile
        import tarfile
        import os
        import pandas as pd
        import requests
        import io
        import pyreadr
        import shutil
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 下载包
            response = requests.get(self.package_url)
            tar_content = io.BytesIO(response.content)
            
            # 解压包
            with tarfile.open(fileobj=tar_content, mode='r:gz') as tar:
                tar.extractall(path=temp_dir)
            
            # 读取.rda文件
            rda_path = os.path.join(temp_dir, self.dataset_path)
            
            # 使用pyreadr读取.rda文件
            result = pyreadr.read_r(rda_path)
            
            # pyreadr返回一个字典，我们取第一个键的值作为DataFrame
            self.data = next(iter(result.values()))
        
        finally:
            # 删除临时目录
            shutil.rmtree(temp_dir)
        
        self.data.reset_index(drop=True, inplace=True)
        
        return self.data

    def load(self):
        pass


class OpenMLDownloader:

    def __init__(self, data_id: int):
        self.data_id = data_id
        self.data = None

    def download(self):
        X, y = fetch_openml(data_id=self.data_id, as_frame=True, return_X_y=True)
        self.data = X.join(y)
        return self.data

    def load(self):
        pass
