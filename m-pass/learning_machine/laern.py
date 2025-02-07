"""
学習マシン
複数のXMLファイルを読み込む。
データをテストデータと学習データに分割する。
学習データを用いてモデルを作成する。
テストデータを用いてモデルを評価する。
ここでは、学習のみを行う
"""

from learning_machine.dataset_maker import DatasetMaker
import pandas as pd


def learn(target_dir: str):
    dataset_maker = DatasetMaker(target_dir)
    test_data, train_data = dataset_maker.make_dataset()
    # 学習データを用いてモデルを作成する。
    model = train_data.model_selection.train_test_split(test_size=0.2)
    print("Learning is done")

    # モデルを保存する。
    model.save("model.h5")
    print("Model is saved")
