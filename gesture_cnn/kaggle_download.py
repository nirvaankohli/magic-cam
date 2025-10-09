import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    "aryarishabh/hand-gesture-recognition-dataset", path="./data", unzip=True
)
