# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# 学習データセットcsvパス設定
csvfile = 'dataset/temp/train.csv'


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # コマンド引数確認
    if len(sys.argv) != 3:
        print('使用法: python3 本ファイル名.py モデルファイル名.h5 CSVファイル名.csv')
        sys.exit()

    # 学習データセットファイル取得
    train_data = load_csv(csvfile)

    # 学習モデルファイルパス取得
    modelfile = sys.argv[1]
    # 学習済ファイルを読み込んでmodelを作成
    rnn_model = keras.models.load_model(modelfile)

    # 入力データファイルパス取得
    plotfile = sys.argv[2]
    # 入力データをロードし変数に格納
    df = pd.read_csv(plotfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    # 特徴量のセットを変数Xに格納
    X = dfv[:, np.array(range(0, n_dfv))]

    # 入力データ標準化
    X = (X - train_data.mean()) / train_data.std()

    # 時系列長定義
    n_rnn = len(X)
    # サンプル数定義
    n_samples = len(train_data) - n_rnn
    # 予測データの格納ベクトルを定義。最初のn_rnn分はXをコピー
    yp = X

    # 予測結果の取得
    for i in range(0, n_samples):
        # 入力データに後続するデータを予測
        y = rnn_model.predict(yp[-n_rnn:].reshape(1, n_rnn, 1))
        #print(y)
        # 出力の最後尾の結果を予測データに格納
        yp = np.append(yp, y[0][0])

    # 予測結果に対し標準化の逆変換
    yp = yp * train_data.std() + train_data.mean()

    # データプロット
    plot_data(train_data, yp)
    # データをCSVに出力
    np.savetxt('result.csv', np.c_[train_data, yp], fmt="%.1f", delimiter=',')


def load_csv(csvfile):
    # csvをロードし、変数に格納
    df = pd.read_csv(csvfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]
    data = dfv[:, np.array((n_dfv-1))]

    return data


def plot_data(y, yp):
    # グラフの軸ラベルを設定
    plt.xlabel('day')
    plt.ylabel('average_temp')

    # x軸データ設定
    n = len(y)
    x = np.linspace(1,n,n)

    # 学習データをプロット
    train, = plt.plot(x, y, marker="o", c='#E69F00')
    # 予測データをプロット
    predict, = plt.plot(x, yp, marker="v", c='#56B4E9')

    # x軸の目盛と表示位置を定義
    x_label = [1, 7, 14, 21, 28]
    plt.xticks(x_label, x_label)

    # グラフの凡例（はんれい）を追加
    plt.legend([train, predict], ['train', 'predict'])

    # グラフを保存
    plt.savefig('result_figure.png')


if __name__ == '__main__':
    main()
