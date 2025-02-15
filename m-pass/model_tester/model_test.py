"""
モデルのテスト
h5形式のファイルをテストする
"""

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json


def evaluate_model(model_path: str):
    # モデルの読み込み（custom_objectsで'mse'を解決）
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    model.summary()
    print(f"[INFO] Model input shape: {model.input_shape}")

    # テストデータの読み込みと基本的な前処理
    wd = os.getcwd()
    test_data_ts = pd.read_csv(os.path.join(wd, "test_data_ts.csv"), index_col=0)

    # インデックスの処理
    test_data_ts.index = pd.to_datetime(test_data_ts.index)  # 既にインデックスは日付型
    test_data_ts = test_data_ts.sort_index()

    # 欠損値の除去
    test_data_ts = test_data_ts.dropna()

    print(f"[DEBUG] Loaded test data shape: {test_data_ts.shape}")
    print(f"[DEBUG] Test data columns: {test_data_ts.columns.tolist()}")

    # ターゲット列の処理
    target_column = "value"
    if target_column not in test_data_ts.columns:
        raise ValueError("Test data must contain the 'value' column")

    # 特徴量の抽出と前処理
    feature_columns = [col for col in test_data_ts.columns if col != target_column]
    X_features = test_data_ts[feature_columns].copy()
    print(f"[DEBUG] Feature columns count: {len(feature_columns)}")

    # 数値変換（エラーの場合は0で補完）
    for col in X_features.columns:
        X_features[col] = pd.to_numeric(X_features[col], errors="coerce").fillna(0)

    # 学習時の特徴量リストの読み込み
    with open(os.path.join(wd, "train_feature_columns.json"), "r") as f:
        train_feature_columns = json.load(f)
    print(f"[DEBUG] Loaded feature columns from JSON: {len(train_feature_columns)}")

    # モデルの期待する特徴量数との調整
    expected_features = model.input_shape[-1]
    current_features = len(train_feature_columns)

    if current_features != expected_features:
        print(
            f"[WARNING] Feature count mismatch. Expected: {expected_features}, Got: {current_features}"
        )
        if current_features < expected_features:
            missing_features = expected_features - current_features
            print(f"[INFO] Adding {missing_features} dummy columns")
            for i in range(expected_features - current_features):
                dummy_col = f"dummy_{i}"
                train_feature_columns.append(dummy_col)

    # 特徴量の順序を学習時と同じに揃える
    X_features = X_features.reindex(columns=train_feature_columns, fill_value=0)
    print(f"[DEBUG] Feature shape after reindex: {X_features.shape}")

    # 次元数の確認
    if X_features.shape[1] != expected_features:
        raise ValueError(
            f"Feature dimension mismatch after reindex. Got {X_features.shape[1]}, expected {expected_features}"
        )

    # ターゲット値の処理
    y_true = pd.to_numeric(test_data_ts[target_column], errors="coerce")

    # スケーリング（特徴量とターゲット）
    scaler_mean = y_true.mean()
    scaler_std = y_true.std()
    y_true = (y_true - scaler_mean) / (scaler_std + 1e-8)  # ゼロ除算を防ぐ

    # 特徴量のスケーリング
    for col in X_features.columns:
        mean = X_features[col].mean()
        std = X_features[col].std()
        if std > 0:  # 標準偏差が0より大きい場合のみスケーリング
            X_features[col] = (X_features[col] - mean) / (std + 1e-8)

    # データ型を float32 に統一（TensorFlowのデフォルト）
    X_features = X_features.astype(np.float32)
    y_true = y_true.astype(np.float32)

    # シーケンス長の取得
    sequence_length = int(model.input_shape[1])

    def sequence_generator():
        batch_size = 256
        n_samples = len(X_features) - sequence_length
        indices = np.arange(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = np.array(
                [
                    X_features.iloc[idx : idx + sequence_length].values
                    for idx in batch_indices
                ]
            )
            y_batch = np.array(
                [y_true.iloc[idx + sequence_length] for idx in batch_indices]
            )

            yield X_batch, y_batch

    # データセットの構築
    test_dataset = tf.data.Dataset.from_generator(
        sequence_generator,
        output_signature=(
            tf.TensorSpec(
                shape=(None, sequence_length, X_features.shape[1]), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    print("[INFO] Starting model evaluation...")

    # モデル評価
    results = model.evaluate(test_dataset, verbose=1)
    metrics_names = model.metrics_names

    print("\nEvaluation Results:")
    for name, value in zip(metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 model_test.py <path_to_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    evaluate_model(model_path)
