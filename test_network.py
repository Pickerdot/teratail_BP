from bpclass import BPneuron, Loss
import matplotlib.pyplot as plt
import numpy as np
from random import random
import matplotlib.pyplot as plt
from IPython.display import clear_output


class test_network:
    def __init__(self, I, H, O, W01_size, W12_size):
        # 重みとバイアスの定義
        W01 = W01_size*np.random.rand(I, H)-W01_size/2
        W12 = W12_size*np.random.rand(H, O)-W12_size/2
        b1 = np.zeros(H)
        b2 = np.zeros(O)

        # モデルの生成
        self.Seccond_layer = BPneuron(W01, b1)
        self.Third_layer = BPneuron(W12, b2)
        self.test_loss_layer = Loss()

        self.loss_memo = []
        self.dx_memo = np.empty(0)
        self.bo_memo = np.empty(0)
        self.dbo_memo = np.empty(0)
        self.W12_memo = np.empty(0)

    def traning(self, test_data, target_data, epoch):
        # 学習に使う配列の決定
        for i in range(epoch):
            test_number = np.random.randint(target_data.shape[0])
            x = np.array([test_data[test_number]])
            t = np.array([target_data[test_number]])

            out1, container1 = self.Seccond_layer.forward(x)
            out2, container2 = self.Third_layer.forward(out1)
            W, b, x, y, z = container1
            W, b, x, y, z = container2

            loss = self.test_loss_layer.forward(out2, t)
            dout = self.test_loss_layer.backward()
            if i % 500 == 0:
                print("loss:", dout, "epoch:", i)
                print("_________")
                print("b:", b)
                print("_________")

            if i % 5000 == 0:
                clear_output()

            self.loss_memo.append(dout[0][0])
            dx, containe = self.Third_layer.backward(dout)
            dinput, containe = self.Seccond_layer.backward(dx)
            self.dx_memo = np.append(self.dx_memo, dx)
            self.bo_memo = np.append(self.bo_memo, b)
            self.W12_memo = np.append(self.W12_memo, W)

        return self.loss_memo

    def dx(self):
        return self.dx_memo

    def bo(self):
        return self.bo_memo

    def W_12(self):
        return self.W12_memo
