"""
データセットを作成する。
"""

import os
import pandas as pd


class DatasetMaker:
    def __init__(self, target_dir: str):
        self.list_of_data = []
        self.test_data = None
        self.train_data = None
        self.dataset = None
        self.target_dir = target_dir

    def make_dataset(self):
        """
        データセットを作成する。
        """
        if self.target_dir is None:
            raise ValueError("target_dir is None")

        list_of_data = []

        # 複数のXMLファイルをtarget_dirから読み込む。
        for file in os.listdir(self.target_dir):
            if file.endswith(".xml"):
                with open(os.path.join(self.target_dir, file), "r", encoding="utf-8") as f:
                    data = f.read()
                    list_of_data.append(data)

        if not list_of_data:
            raise ValueError("No XML files found in target_dir")

        # すべてのデータをDataFrameに格納（1列 "xml_data" として）
        dataset = pd.DataFrame(list_of_data, columns=["xml_data"])
        self.dataset = dataset  # 必要に応じて保持
        test_data = dataset.sample(frac=0.2, random_state=42)
        train_data = dataset.drop(test_data.index)
        return test_data, train_data
