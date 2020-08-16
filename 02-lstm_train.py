# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time


# 時系列長
n_rnn = 7
# バッチサイズ設定
n_bs = 4

# 中間層ニューロン数設定
n_units = 20
# 出力層ニューロン数設定
n_out = 1
# ドロップアウト率
r_dropout = 0.0
# エポック数
nb_epochs = 1000

# 学習データセットcsvパス設定
csvfile = 'dataset/temp/train.csv'


def main():
    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'
    # 時間計測開始
    start = time.time()

    # コマンド引数確認
    if len(sys.argv) != 2:
        print('使用法: python3 本ファイル名.py モデルファイル名.h5')
        sys.exit()
    # 学習モデルファイルパス取得
    savefile = sys.argv[1]
    # データセットファイル読み込み
    data = load_csv(csvfile)
    # サンプル数
    n_samples = len(data) - n_rnn

    # 入力データ定義
    x = np.zeros((n_samples, n_rnn))
    # 正解データ定義
    t = np.zeros((n_samples,))
    for i in range(0, n_samples):
        x[i] = data[i:i + n_rnn]
        # 正解データは1単位時刻後の値
        t[i] = data[i + n_rnn]
    #print(x)
    #print(t)

    # Keras向けにxとtを整形
    x = x.reshape(n_samples, n_rnn, 1)
    t = t.reshape(n_samples, 1)
    #print(x.shape)
    #print(t.shape)

    # データシャッフル
    p = np.random.permutation(n_samples)
    x = x[p]
    t = t[p]

    # モデル作成(既存モデルがある場合は読み込んで再学習。なければ新規作成)
    if os.path.exists(savefile):
        print('モデル再学習')
        rnn_model = keras.models.load_model(savefile)
    else:
        print('モデル新規作成')
        rnn_model = rnn_model_maker(n_samples, n_out)

    # モデル構造の確認
    rnn_model.summary()

    # モデルの学習
    history = rnn_model.fit(x, t, epochs=nb_epochs, validation_split=0.1, batch_size=n_bs, verbose=2)
    # 学習結果を保存
    rnn_model.save(savefile)

    # 学習所要時間の計算、表示
    process_time = (time.time() - start) / 60
    print('process_time = ', process_time, '[min]')

    # 損失関数の時系列変化をグラフ表示
    plot_loss(history)


def load_csv(csvfile):
    # csvをロードし、変数に格納
    df = pd.read_csv(csvfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    data = dfv[:, np.array((n_dfv-1))]
    #print(data)

    # データの標準化
    data = (data - data.mean()) / data.std()
    #print(data)

    return data


def rnn_model_maker(n_samples, n_out):
    # 3層RNN(リカレントネットワーク)を定義
    model = Sequential()
    # 中間層(RNN)を定義
    model.add(LSTM(units=n_units, input_shape=(n_rnn, 1), dropout=r_dropout, return_sequences=False))
    # 出力層を定義(ニューロン数は1個)
    model.add(Dense(units=n_out, activation='linear'))
    # 回帰学習モデル作成
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # モデルを返す
    return model


def plot_loss(history):
    # 損失関数のグラフの軸ラベルを設定
    plt.xlabel('time step')
    plt.ylabel('loss')

    # グラフ縦軸の範囲を0以上と定める
    plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))

    # 損失関数の時間変化を描画
    val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
    loss, = plt.plot(history.history['loss'], c='#E69F00')

    # グラフの凡例（はんれい）を追加
    plt.legend([loss, val_loss], ['loss', 'val_loss'])

    # 描画したグラフを表示
    #plt.show()

    # グラフを保存
    plt.savefig('train_figure.png')


if __name__ == '__main__':
    main()
