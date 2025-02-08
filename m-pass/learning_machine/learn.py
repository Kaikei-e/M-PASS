from learning_machine.dataset_maker import DatasetMaker
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # 進捗表示用

# タスク用の関数をグローバルレベルに移動
def process_chunk(input_data: pd.DataFrame, output_data: pd.Series, sequence_length: int, start: int, chunk_size: int, n_samples: int):
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
        X_sample = input_data[i: i + sequence_length].values
        y_sample = output_data.iloc[i + sequence_length]
        X_chunk.append(X_sample)
        y_sample = np.array([y_sample])
        y_chunk.append(y_sample)
    return X_chunk, y_chunk


def create_sequences(input_data, output_data, sequence_length):
    """
    指定されたシーケンス長に基づいて時系列の入力シーケンスとターゲットを作成する関数。
    ベクトル化処理に加えて、各工程の処理時間や進捗をログ出力することでボトルネックを可視化する。

    Parameters:
        input_data (pd.DataFrame): 特徴量のデータフレーム。
        output_data (pd.Series): 対応するターゲット値のシリーズ。
        sequence_length (int): シーケンスの長さ。

    Returns:
        X (np.ndarray): シーケンス格納用配列。shape=(n_samples, sequence_length, n_features)
        y (np.ndarray): ターゲット値配列。shape=(n_samples,)
    """
    import time

    # --- ステップ1: NumPy 配列に変換 ---
    t0 = time.time()
    input_array = (
        input_data.to_numpy() if hasattr(input_data, "to_numpy") else np.asarray(input_data)
    )
    output_array = (
        output_data.to_numpy() if hasattr(output_data, "to_numpy") else np.asarray(output_data)
    )
    t1 = time.time()
    print(f"[DEBUG] Conversion to NumPy arrays took {t1 - t0:.3f} seconds.")

    N = input_array.shape[0]
    if N <= sequence_length:
        raise ValueError("データ数がシーケンス長より少ないため、シーケンス作成ができません。")

    # --- ステップ2: sliding_window_view によるシーケンス窓の取得 ---
    t2 = time.time()
    X_windows = np.lib.stride_tricks.sliding_window_view(input_array, window_shape=sequence_length, axis=0)
    t3 = time.time()
    print(f"[DEBUG] Sliding window view creation took {t3 - t2:.3f} seconds.")

    # sliding_window_view の結果は view なので、全体のコピーが遅延する可能性がある
    # X_windows の shape は (N - sequence_length + 1, sequence_length, n_features) となる
    n_total = X_windows.shape[0] - 1
    print(f"[DEBUG] Total sequences to copy: {n_total}")

    # --- ステップ3: チャンクごとにコピー (コピー負荷の可視化のため進捗バー付き) ---
    chunk_size = 100000  # このサイズはメモリ状況に応じて調整可能
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    X_chunks = []
    t4 = time.time()
    for i in tqdm(range(n_chunks), desc="Copying sequence chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_total)
        # 各チャンクごとにビューからコピー
        X_chunks.append(X_windows[start_idx: end_idx].copy())
    X = np.concatenate(X_chunks, axis=0)
    t5 = time.time()
    print(f"[DEBUG] Copying {n_chunks} chunks took {t5 - t4:.3f} seconds.")

    # --- ステップ4: ターゲット値の抽出 ---
    t6 = time.time()
    y = output_array[sequence_length:]
    t7 = time.time()
    print(f"[DEBUG] Extracting target values took {t7 - t6:.3f} seconds.")

    total_time = t7 - t0
    print(f"[DEBUG] Total create_sequences time: {total_time:.3f} seconds.")
    return X, y


def create_sequences_generator(input_data, output_data, sequence_length, batch_size):
    """
    バッチごとに連続したシーケンスを生成するジェネレータ関数です。
    
    ・入力データ、出力データは事前にNumPy配列に変換し、
      np.lib.stride_tricks.sliding_window_view により view を取得。
    ・バッチごとに np.ascontiguousarray を用いて連続メモリブロックに変換することで、
      下流のTensorFlow処理を高速化します。
    """
    input_array = input_data.to_numpy() if hasattr(input_data, "to_numpy") else np.asarray(input_data)
    print(f"[DEBUG] Input array shape: {input_array.shape}")
    output_array = output_data.to_numpy() if hasattr(output_data, "to_numpy") else np.asarray(output_data)
    print(f"[DEBUG] Output array shape: {output_array.shape}")
    
    total_sequences = input_array.shape[0] - sequence_length
    print(f"[DEBUG] Total sequences available: {total_sequences}")
    
    # sliding_window_view はデータの view を返す（コピーは発生しない）
    X_view = np.lib.stride_tricks.sliding_window_view(input_array, window_shape=sequence_length, axis=0)
    
    for start in tqdm(range(0, total_sequences, batch_size), desc="Generating sequence batches"):
        end = min(start + batch_size, total_sequences)
        X_batch = np.ascontiguousarray(X_view[start:end])
        y_batch = output_array[start + sequence_length : end + sequence_length]
        yield X_batch, y_batch


def learn(target_dir):
    """
    モデル学習のフロー：
      1. DatasetMaker でXMLファイルからデータをストリーミング処理し、DataFrameを作成
      2. 前処理（One-Hot Encoding、時系列設定、補完処理など）
      3. create_sequences_generator により、バッチ単位でシーケンス生成
      4. LSTM モデルの定義・学習
    """
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

    # カテゴリ変数のOne-Hot Encoding
    categorical_columns = ["type", "unit"]
    print(f"[INFO] One-Hot Encoding on columns: {categorical_columns}")
    train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns)
    test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns)

    # 時系列処理：endDate をインデックスに設定
    train_data_ts = train_data_encoded.set_index("endDate")
    test_data_ts = test_data_encoded.set_index("endDate")
    train_data_ts = train_data_ts.set_index(pd.to_datetime(train_data_ts.index))
    test_data_ts = test_data_ts.set_index(pd.to_datetime(test_data_ts.index))

    # target カラムの補完（数値変換、欠損値補完）
    target_column = "value"
    train_data_ts[target_column] = pd.to_numeric(train_data_ts[target_column], errors="coerce")
    test_data_ts[target_column] = pd.to_numeric(test_data_ts[target_column], errors="coerce")
    train_data_ts = train_data_ts.infer_objects().interpolate(method="linear")
    test_data_ts = test_data_ts.infer_objects().interpolate(method="linear")

    # 特徴量は target カラム以外の全カラムを利用
    feature_columns = [col for col in train_data_ts.columns if col != target_column]
    sequence_length = 5
    print(f"[INFO] Sequence generation with sequence_length = {sequence_length}")
    print(f"[INFO] Number of feature columns: {len(feature_columns)}")

    # バッチ生成用のジェネレータ作成（メモリ効率向上のため下流に連続配列として供給）
    batch_size = 100000  # 必要に応じて調整
    train_gen = create_sequences_generator(
        train_data_ts[feature_columns].fillna(train_data_ts[feature_columns].mean()),
        train_data_ts[target_column].fillna(train_data_ts[target_column].mean()),
        sequence_length,
        batch_size
    )

    total_sequences = len(train_data_ts) - sequence_length
    steps_per_epoch = (total_sequences + batch_size - 1) // batch_size
    print(f"[INFO] Steps per epoch: {steps_per_epoch}")

    # LSTMモデルの定義（シンプルな構造の例）
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            units=50,
            activation="relu",
            input_shape=(sequence_length, len(feature_columns))
        ),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    epochs = 10
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )

    print("[INFO] Model training complete.")
    # ここで予測や評価も実施可能です。

