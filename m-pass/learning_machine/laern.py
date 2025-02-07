"""
学習マシン
複数のXMLファイルを読み込む。
データをテストデータと学習データに分割する。
学習データを用いてモデルを作成する。
テストデータを用いてモデルを評価する。
ここでは、学習のみを行う
"""

from learning_machine.dataset_maker import DatasetMaker # DatasetMaker のインポート
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np # NumPy のインポート
from sklearn.preprocessing import MinMaxScaler # 正規化用
from tensorflow.keras.models import Sequential # Keras モデル
from tensorflow.keras.layers import LSTM, Dense # Keras レイヤー
import matplotlib.pyplot as plt # 学習曲線描画用

def learn(target_dir: str):
    dataset_maker = DatasetMaker(target_dir)
    test_data_original, train_data_original = dataset_maker.make_dataset() # データセット作成
    # 学習データを用いて train_test_split を実行する。(元データを保持しておく)
    train_data, val_data = train_test_split(train_data_original, test_size=0.2, random_state=42) # train_data をさらに分割して検証データを作成
    print("Training, validation, and test split complete")

    # **データ前処理 (正規化)**
    scaler = MinMaxScaler() # 正規化スケーラー
    scaler.fit(train_data) # train_data で fit
    train_data_scaled = scaler.transform(train_data) # 正規化
    val_data_scaled = scaler.transform(val_data) # 検証データも正規化 (train_data のスケーラーを使用)
    test_data_scaled = scaler.transform(test_data_original) # テストデータも正規化 (train_data のスケーラーを使用)

    # **時系列データ作成**
    sequence_length = 24 * 60 * 10  # 例: 過去24時間
    X_train, y_train = create_sequences(train_data_scaled, sequence_length)
    X_val, y_val = create_sequences(val_data_scaled, sequence_length) # 検証データもsequence作成
    X_test, y_test = create_sequences(test_data_scaled, sequence_length)

    # **LSTM モデル構築**
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=y_train.shape[2])) # 出力層は特徴量数に合わせる
    model.compile(optimizer='adam', loss='mse') # 損失関数は回帰なので mse

    model.summary() # モデル概要を表示

    # **モデル学習**
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val)) # 検証データを指定

    # **学習曲線描画**
    plot_learning_curve(history)

    # **モデル評価**
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # **モデル保存 (オプション)**
    model.save("mood_predictor_model.keras") # Kerasの標準形式で保存
    print("Model is saved as mood_predictor_model.keras")


def create_sequences(data, sequence_length):
    """時系列データをLSTM入力形式に変換する関数 (DatasetMaker の出力に合わせて調整が必要な場合あり)"""
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = data[i+sequence_length:i+sequence_length+1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def plot_learning_curve(history):
    """学習曲線を描画する関数"""
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss') # 検証データ損失もプロット
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve') # タイトル追加
    plt.legend()
    plt.show()
