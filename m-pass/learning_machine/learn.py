from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from learning_machine.dataset_maker import DatasetMaker
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # 進捗表示用
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

# タスク用の関数をグローバルレベルに移動
def process_chunk(
    input_data: pd.DataFrame,
    output_data: pd.Series,
    sequence_length: int,
    start: int,
    chunk_size: int,
    n_samples: int,
):
    """
    指定されたインデックスからchunk_size分のシーケンスを作成する関数。

    Parameters:
        input_data (pd.DataFrame): 特徴量のDataFrame。
        output_data (pd.Series): 目的変数のSeries。
        sequence_length (int): シーケンスの長さ。
        start (int): チャンク開始インデックス。
        chunk_size (int): チャンクサイズ。
        n_samples (int): 全体のシーケンス数。

    Returns:
        Tuple (X_chunk, y_chunk): 部分シーケンスと対応するターゲットのリスト。
    """
    X_chunk = []
    y_chunk = []
    end = min(start + chunk_size, n_samples)
    for i in range(start, end):
        X_sample = input_data[i : i + sequence_length].values
        y_sample = output_data.iloc[i + sequence_length]
        X_chunk.append(X_sample)
        y_sample = np.array([y_sample])
        y_chunk.append(y_sample)
    return X_chunk, y_chunk


def create_sequences(input_data, output_data, sequence_length):
    """
    sliding_window_view を使い、ビューのまま処理するバージョン（コピー不要の場合）
    """
    input_array = input_data.to_numpy()
    output_array = output_data.to_numpy()

    N = input_array.shape[0]
    if N <= sequence_length:
        raise ValueError(
            "データ数がシーケンス長より少ないため、シーケンス作成ができません。"
        )

    X = np.lib.stride_tricks.sliding_window_view(
        input_array, window_shape=sequence_length, axis=0
    )[:-1]  # 最後の要素はターゲットがないので除外
    y = output_array[sequence_length:]

    return X, y

def create_sequences_generator(input_data, output_data, sequence_length, batch_size):
    """
    メモリ効率の良いシーケンス生成ジェネレータ (with ステートメントを使用)
    """
    input_array = input_data.to_numpy()
    output_array = output_data.to_numpy()
    total_samples = len(input_array) - sequence_length
    n_features = input_array.shape[1]

    # マルチスレッドプールの作成
    n_workers = min(os.cpu_count(), 16)

    def process_batch(start_idx):
        """バッチ処理用の内部関数"""
        end_idx = min(start_idx + batch_size, total_samples)
        X_batch = []
        y_batch = []
        for i in range(start_idx, end_idx):
            X_batch.append(input_array[i:i + sequence_length])
            y_batch.append(output_array[i + sequence_length])

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    # バッチインデックスの生成
    batch_starts = range(0, total_samples, batch_size)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for start_idx in tqdm(batch_starts, desc="Generating sequences"):
            future = executor.submit(process_batch, start_idx)
            X_batch, y_batch = future.result()
            yield X_batch, y_batch

def learn(target_dir):
    """
    最適化されたモデル学習のフロー
    """
    wd = os.getcwd()

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    print("[INFO] Dataset creation started...")
    dataset_maker = DatasetMaker(target_dir)
    test_data, train_data = dataset_maker.make_dataset()

    if train_data.empty:
        raise ValueError("Train dataset is empty. XMLデータの内容を確認してください。")
    if test_data.empty:
        print("[WARNING] Test dataset is empty.")

    print("[INFO] Dataset state after creation:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")

    # データの前処理
    train_data_ts, test_data_ts, scaler = clean_data(train_data, test_data)
    feature_columns = train_data_ts.columns
    target_column = "value"
    sequence_length = 5000

    # バッチサイズの設定
    batch_size = 128

    # TensorFlowのメモリ使用量を制限しない
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                print(f"[INFO] Setting memory growth for GPU: {gpu}")
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")

    # データセットをTensorFlow形式に変換し、プリフェッチを使用
    train_gen = create_sequences_generator(
        train_data_ts[feature_columns],
        train_data_ts[target_column],
        sequence_length,
        batch_size,
    )
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    print("[INFO] Dataset creation started...")
    dataset_maker = DatasetMaker(target_dir)
    test_data, train_data = dataset_maker.make_dataset()

    if train_data.empty or test_data.empty:  # 両方とも空でないかチェック
        raise ValueError(
            "Train or Test dataset is empty. XMLデータの内容を確認してください。"
        )

    print("[INFO] Dataset state after creation:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")

    # ジェネレータからデータセットを作成
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(None, sequence_length, len(feature_columns)), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    test_gen = create_sequences_generator(
        test_data_ts[feature_columns],
        test_data_ts[target_column],
        sequence_length,
        batch_size,
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(None, sequence_length, len(feature_columns)), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    # モデルの定義（より効率的な構成）
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                units=32,
                activation="relu",
                input_shape=(sequence_length, len(feature_columns)),
                return_sequences=False,
                dropout=0.3,  # オーバーフィッティング防止
                kernel_regularizer=regularizers.l2(0.01),

            ),
            tf.keras.layers.BatchNormalization(),  # 学習の安定化
            tf.keras.layers.Dense(units=1),
            tf.keras.layers.Dropout(0.3),
        ]
    )

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # 最適化設定
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"],  # 平均絶対誤差も追跡
    )

    # コールバックの設定
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),  # val_loss を監視
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),  # val_loss を監視
    ]

    # モデルの学習 (validation_data を指定)
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=test_dataset,  # 検証データを渡す
        callbacks=callbacks,
        verbose=1,
    )

    wd = os.getcwd()

    # モデルを保存
    model_save_path = os.path.join(wd, "model_tester", "model.h5")
    print(f"[INFO] Saving model to {model_save_path}")
    model.save(model_save_path)

    # 学習履歴も保存（オプション）
    history_save_path = os.path.join(wd, "model_tester", "training_history.npy")
    np.save(history_save_path, history.history)

    print("[INFO] Model and history saved successfully")

    return model, history



def preprocess_features(data_ts: pd.DataFrame, scaler: Optional[StandardScaler] = None, target_column: str = "value") -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    特徴量の前処理を行う関数 (学習データとテストデータで共通化)
    """
    feature_columns = [col for col in data_ts.columns if col != target_column]
    X_features = data_ts[feature_columns].copy()
    y_values = pd.to_numeric(data_ts[target_column], errors='coerce')

    for col in X_features.columns:
        X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0)

    # 学習時のみ Scaler を作成・適用、テスト時は学習時の Scaler を使用
    if scaler is None:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_values.values.reshape(-1, 1)).flatten()  # fit_transform を使用
    else:
        y_scaled = scaler.transform(y_values.values.reshape(-1, 1)).flatten()

    for col in X_features.columns:
        mean = X_features[col].mean()
        std = X_features[col].std()
        if std > 0:
            X_features[col] = (X_features[col] - mean) / (std + 1e-8)

    return X_features.astype(np.float32), y_scaled.astype(np.float32), scaler

def clean_data(train_data, test_data):
    """
    データの前処理を行う関数
    """
    print("[INFO] Starting data preprocessing...")

    # Define target column
    target_column = "value"

    # One-Hot Encoding for categorical columns
    categorical_columns = ["type", "unit"]
    print(f"[INFO] One-Hot Encoding on columns: {categorical_columns}")
    train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns)
    test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns)

    # Set datetime index
    train_data_ts = train_data_encoded.set_index("endDate")
    test_data_ts = test_data_encoded.set_index("endDate")
    train_data_ts = train_data_ts.set_index(pd.to_datetime(train_data_ts.index))
    test_data_ts = test_data_ts.set_index(pd.to_datetime(test_data_ts.index))
    train_data_ts = train_data_ts.sort_index()
    test_data_ts = test_data_ts.sort_index()

    # Remove rows with missing values
    train_data_ts = train_data_ts.dropna()
    test_data_ts = test_data_ts.dropna()

    # Process training data
    print("[INFO] Processing training data...")
    X_train, y_train, scaler = preprocess_features(train_data_ts, None, target_column)
    train_data_ts = pd.concat(
        [X_train, pd.Series(y_train, index=X_train.index, name=target_column)], axis=1
    )

    # Process test data
    print("[INFO] Processing test data...")
    X_test, y_test, _ = preprocess_features(test_data_ts, None, target_column)
    test_data_ts = pd.concat(
        [X_test, pd.Series(y_test, index=X_test.index, name=target_column)], axis=1
    )

    # テストデータの前処理: 学習データ側の全カラムに合わせる
    train_feature_columns = [
        col for col in train_data_ts.columns if col != target_column
    ]

    # Save feature columns for later use in testing
    wd = os.getcwd()
    with open(os.path.join(wd, "model_tester", "train_feature_columns.json"), "w") as f:
        json.dump(train_feature_columns, f)

    # 学習データと同じ順序・カラム数に揃える（余分なカラムは削除し、不足分は0で埋める）
    test_data_ts = test_data_ts.reindex(
        columns=(train_feature_columns + [target_column]), fill_value=0
    )

    # テストデータの保存
    test_data_ts.to_csv(
        os.path.join(wd, "model_tester", "test_data_ts.csv"), index=True
    )

    print(
        f"[INFO] Final shapes - Train: {train_data_ts.shape}, Test: {test_data_ts.shape}"
    )
    return train_data_ts, test_data_ts, scaler
