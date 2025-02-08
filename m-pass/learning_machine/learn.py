from learning_machine.dataset_maker import DatasetMaker
import os
import pandas as pd
import numpy as np
import tensorflow as tf  # TensorFlow をインポート  ※TensorFlow がインストールされている必要あり
from sklearn.metrics import mean_squared_error



def learn(
    target_dir,
):
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    dataset_maker = DatasetMaker(target_dir)

    print("Dataset creation started...")
    test_data_original, train_data_original = (
        dataset_maker.make_dataset()
    )  # make_dataset() が分割されたデータセットを返すように変更

    if train_data_original.empty:  # 学習データセットが空の場合はエラーとする (変更なし)
        raise ValueError(
            "Train dataset is empty. Please check your XML data and DatasetMaker implementation."
        )
    if test_data_original.empty:  # テストデータセットが空の場合は警告とする (変更なし)
        print(
            "Warning: Test dataset is empty. Please check your XML data or consider reducing test_size in DatasetMaker."
        )

    print("\n[Debug] Datasetの状態 (dataset_maker.make_dataset() 直後):")
    print("train_data_original.empty:", train_data_original.empty)
    print("test_data_original.empty:", test_data_original.empty)
    print("train_data_original.shape:", train_data_original.shape)
    print("test_data_original.shape:", test_data_original.shape)
    print("train_data_original.head():\n", train_data_original.head())
    print(
        "test_data_original.head():\n", test_data_original.head()
    )  # テストデータセットの head() も出力 # 変更
    print("--------------------------------------------------")

    # ★★★  カテゴリデータ One-Hot Encoding (type, unit) ★★★ (変更)
    print(
        "\n[Debug] データ型確認 (One-Hot Encoding 前):"
    )  # One-Hot Encoding 前のデータ型確認
    print(train_data_original.dtypes)  # 学習データセットのデータ型を出力
    print(test_data_original.dtypes)  # テストデータセットのデータ型を出力

    # カテゴリデータ列を特定 (例: 'type', 'unit' 列,  必要に応じて列名を追加・修正)
    categorical_columns = ["type", "unit"]  # 'unit' 列も追加！ # 変更

    print(
        "\n[Debug] One-Hot Encoding 対象列:", categorical_columns
    )  # One-Hot Encoding 対象列をログ出力

    # 学習データとテストデータで One-Hot Encoding を実行
    train_data_encoded = pd.get_dummies(
        train_data_original, columns=categorical_columns
    )
    test_data_encoded = pd.get_dummies(test_data_original, columns=categorical_columns)

    print(
        "\n[Debug] One-Hot Encoding 後のデータ形状:"
    )  # One-Hot Encoding 後のデータ形状をログ出力
    print(
        "train_data_encoded.shape:", train_data_encoded.shape
    )  # 学習データセットの shape を出力
    print(
        "test_data_encoded.shape:", test_data_encoded.shape
    )  # テストデータセットの shape を出力

    print(
        "\n[Debug] One-Hot Encoding 後のデータ型:"
    )  # One-Hot Encoding 後のデータ型を出力
    print(train_data_encoded.dtypes)  # 学習データセットのデータ型を出力
    print(test_data_encoded.dtypes)  # テストデータセットのデータ型を出力

    print(
        "\n[Debug] One-Hot Encoding 後の学習データ (train_data_encoded.head()):"
    )  # One-Hot Encoding 後の学習データ head() を出力
    print(train_data_encoded.head())
    print(
        "\n[Debug] One-Hot Encoding 後のテストデータ (test_data_encoded.head()):"
    )  # One-Hot Encoding 後のテストデータ head() を出力
    print("--------------------------------------------------")

    # ★★★  LSTM モデル  ★★★
    print("\n[Debug] LSTM モデル 学習・評価 開始")  # LSTM モデル 学習・評価 開始ログ

    # 1. 時系列データとして整形 (endDate をインデックスに設定)
    train_data_ts = train_data_encoded.set_index(
        "endDate"
    )  # 学習データで 'endDate' をインデックスに設定
    test_data_ts = test_data_encoded.set_index(
        "endDate"
    )  # テストデータで 'endDate' をインデックスに設定

    # インデックスを DatetimeIndex に変換 (もし 'endDate' 列が文字列型の場合)
    train_data_ts = train_data_ts.set_index(pd.to_datetime(train_data_ts.index))
    test_data_ts = test_data_ts.set_index(pd.to_datetime(test_data_ts.index))

    # ★★★  value 列を数値型 (float64) に明示的に変換 (エラー回避応急処置) ★★★
    target_column = "value"  # 予測対象列 (例: 'value' 列)
    train_data_ts[target_column] = pd.to_numeric(
        train_data_ts[target_column], errors="coerce"
    )  # 数値に変換、エラーは NaN に
    test_data_ts[target_column] = pd.to_numeric(
        test_data_ts[target_column], errors="coerce"
    )  # テストデータも同様に変換

    print(
        "\n[Debug] LSTM モデル入力データ型 (数値型変換後):"
    )  # データ型確認ログ (数値型変換後)
    print(
        "train_data_ts[target_column].dtype:\n", train_data_ts[target_column].dtype
    )  # データ型を出力
    print(
        "train_data_ts[target_column].head():\n",
        train_data_ts[target_column].head(),
    )  # データの中身 (先頭5行) を出力
    print("--------------------------------------------------")
    # ★★★  数値型変換処理 (ここまで) ★★★

    # 欠損値処理 (例: 線形補間) - 必要に応じて
    train_data_ts = train_data_ts.interpolate(
        method="linear"
    )  # 学習データの欠損値を線形補間
    test_data_ts = test_data_ts.interpolate(
        method="linear"
    )  # テストデータの欠損値を線形補間

    print(
        "\n[Debug] LSTM モデル入力データ型 (interpolate() 適用後):"
    )  # データ型確認ログ (interpolate() 適用後)
    print(
        "train_data_ts[target_column].dtype:\n", train_data_ts[target_column].dtype
    )  # データ型を出力
    print(
        "train_data_ts[target_column].head():\n",
        train_data_ts[target_column].head(),
    )  # データの中身 (先頭5行) を出力
    print("--------------------------------------------------")
    # ★★★ デバッグログ追加 (ここまで) ★★★

    # 2. データシーケンス作成 (LSTM モデル入力用)  ★★★ (追加)
    sequence_length = (
        20  #  LSTM に入力する過去のデータ数 (シーケンス長) を定義 (例: 20)
    )
    target_column = "value"  # 予測対象列 (例: 'value' 列)
    feature_columns = [
        col for col in train_data_ts.columns if col != target_column
    ]  # 特徴量として使用する列 (target_column 以外全て)

    print(
        "\n[Debug] LSTM モデル データシーケンス作成 (sequence_length):", sequence_length
    )  # データシーケンス作成設定ログ
    print("[Debug] LSTM モデル 特徴量列:", feature_columns)  # 特徴量列ログ

    # 学習データとテストデータからシーケンスを作成
    X_train, y_train = create_sequences(
        train_data_ts[feature_columns].fillna(
            train_data_ts[feature_columns].mean()
        ),  # 特徴量列の欠損値を平均値で補完
        train_data_ts[target_column].fillna(
            train_data_ts[target_column].mean()
        ),  # ターゲット列の欠損値を平均値で補完
        sequence_length,
    )
    X_test, y_test = create_sequences(
        test_data_ts[feature_columns].fillna(
            test_data_ts[feature_columns].mean()
        ),  # 特徴量列の欠損値を平均値で補完
        test_data_ts[target_column].fillna(
            test_data_ts[target_column].mean()
        ),  # ターゲット列の欠損値を平均値で補完
        sequence_length,
    )

    print("\n[Debug] LSTM モデル データシーケンス形状:")  # データシーケンス形状ログ
    print("X_train.shape:", X_train.shape)  # 学習データ入力シーケンスの形状
    print("y_train.shape:", y_train.shape)  # 学習データターゲットの形状
    print("X_test.shape:", X_test.shape)  # テストデータ入力シーケンスの形状
    print("y_test.shape:", y_test.shape)  # テストデータターゲットの形状
    print("--------------------------------------------------")

    # 3. LSTM モデル構築  ★★★ (追加)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                units=50,
                activation="relu",
                input_shape=(X_train.shape[1], X_train.shape[2]),
            ),  # LSTM層 (units は LSTM ユニット数, input_shape は入力シーケンスの形状)
            tf.keras.layers.Dense(
                units=1
            ),  # 出力層 (units=1 は 1次元の値を予測するため)
        ]
    )

    # 4. モデルコンパイル  ★★★ (追加)
    model.compile(
        optimizer="adam", loss="mse"
    )  # Optimizer は Adam, 損失関数は MSE を使用

    print("\n[Debug] LSTM モデル モデル構造:")  # モデル構造ログ
    model.summary()  # モデル構造を summary で出力

    # 5. モデル学習  ★★★ (追加)
    epochs = 10  # 学習エポック数 (調整可能)
    batch_size = 32  # バッチサイズ (調整可能)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )  # モデル学習 (verbose=0 で学習ログ非表示)

    print("\n[Debug] LSTM モデル 学習履歴:")  # 学習履歴ログ
    print(history.history)  # 学習履歴 (損失関数の値など) を出力

    # 6. 予測  ★★★ (追加)
    predictions = model.predict(
        X_test, verbose=0
    )  # テストデータで予測 (verbose=0 で予測ログ非表示)

    print(
        "\n[Debug] LSTM モデル 予測結果 (predictions[:5]):"
    )  # LSTM モデル 予測結果 (predictions[:5]) ログ
    print(predictions[:5])  # 予測値の先頭5行を出力

    # 7. 評価 (MSE)  ★★★ (変更)
    mse = mean_squared_error(
        y_test, predictions
    )  # テストデータの実測値 (y_test) と予測値 (predictions) を MSE で評価 # 変更
    print(f"\nMean Squared Error (LSTM): {mse}")  # MSE を出力 # 変更

    print("\n[Debug] LSTM モデル 学習・評価 完了")  # LSTM モデル 学習・評価 完了ログ


def create_sequences(
    input_data, output_data, sequence_length
):  # データシーケンスを作成する関数 (変更なし)
    X, y = [], []
    for i in range(len(input_data) - sequence_length):
        X.append(input_data[i : i + sequence_length].values)
        y.append(output_data.iloc[i + sequence_length])  # output_data は Series を想定
    return np.array(X), np.array(y)
