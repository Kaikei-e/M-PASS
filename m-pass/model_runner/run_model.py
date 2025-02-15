import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys


def run_ml_model(model_path: str, test_data_csv_path: str):  # 引数名をより明確に変更
    """
    機械学習モデルをロードし、テストデータで予測を実行し、結果をCSVファイルに保存します。

    Args:
        model_path (str): TensorFlowモデル（.h5ファイル）のパス。
        test_data_csv_path (str): テストデータCSVファイルのパス。

    Returns:
        str: 結果が保存されたCSVファイルのパス。エラーが発生した場合はNoneを返します。
    """
    try:
        # モデルの読み込み
        model = tf.keras.models.load_model(model_path)

        # テストデータの読み込み (CSVファイルのパスからDataFrameへ)
        test_data_df = pd.read_csv(test_data_csv_path)

        # **重要**: モデルの入力として適切な形式にデータを準備する必要があります。
        #        以下の部分は、実際のデータとモデルの入力形式に合わせて修正が必要です。
        #        ここでは、DataFrame全体をNumPy配列に変換する例を示しています。
        test_data = test_data_df.values

        # モデルの予測を実行
        predictions = model.predict(test_data)

        # 予測結果をDataFrameに変換 (必要に応じて列名を設定)
        predictions_df = pd.DataFrame(
            predictions, columns=["prediction"]
        )  # 列名を 'prediction' に設定 (例)

        # 予測結果をCSVファイルに保存
        wd = os.getcwd()
        result_csv_path = os.path.join(wd, "results.csv")
        predictions_df.to_csv(result_csv_path, index=False)

        return result_csv_path  # 保存したCSVファイルのパスを返す

    except FileNotFoundError:
        print(f"エラー: モデルファイルまたはテストデータCSVファイルが見つかりません。")
        return None
    except Exception as e:
        print(f"エラー: モデル実行中に予期せぬエラーが発生しました: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 run_model.py <model_path> <test_data_csv_path>"
        )  # 引数名を修正
        sys.exit(1)

    model_path = sys.argv[1]
    test_data_csv_path = sys.argv[2]  # 変数名を修正

    result_path = run_ml_model(model_path, test_data_csv_path)  # 変数名を修正

    if result_path:
        print(f"予測結果は以下のファイルに保存されました: {result_path}")
    else:
        print("予測処理に失敗しました。エラーログを確認してください。")
