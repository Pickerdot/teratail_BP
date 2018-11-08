from optimizer import optimizer_SGD
import numpy as np
from functions import sigmoid, sigmoid_back


lr = 0.01


class Loss:
    def __init__(self):
        self.Loss = None
        self.dout = None

    def forward(self, out, t):
        self.Loss = 1/2 * np.sum((out - t)**2)
        self.dout = out - t
        return self.Loss

    def backward(self):
        return self.dout


class BPneuron:
    def __init__(self, W, b):
        # 引数として受けた重みとバイアスをself.aramsに格納
        self.params = [W, b]
        # 更新前に勾配をまとめてオプティマイザーに送るための入れ物（中身はparamsに対応している必要あり）
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        # クラス外へ中身を持っていくための入れ物
        self.container = np.empty(0)

    def forward(self, x):
        # クラスの初期化時に格納した重みとバイアスの取り出し
        W, b = self.params
        # yはニューロン内部の値
        y = np.dot(x, W)+b
        # Zが出力
        z = sigmoid(y)
        self.container = [W, b, x, y, z]
        return z, self.container

    def backward(self, dz):
        W, b, x, y, z = self.container
        # 出力部の逆伝搬（シグモイド版）
        dy = sigmoid_back(z, dz)
        db = dy
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)

        # self.gradsに更新に行かう勾配を格納
        self.grads[0][...] = dW
        self.grads[1][...] = db

        # オプティマイザーによりself.paramsの値を更新
        self.params = optimizer_SGD(lr, self.params, self.grads)
        # すべての結果をself.containerに格納
        self.container = [dy, db, dW, dx]

        return dx, self.container
