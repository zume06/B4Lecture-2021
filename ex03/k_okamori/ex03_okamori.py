import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# 最小二乗法を実現するクラス(1変数)


class LeastSqueares():

    # コンストラクタ
    #   data_observed: csv から得たデータ
    #   degree: 近似に用いる次数
    #   nc: 正則化
    def __init__(self, data_observed, degree, nc):
        self.data_observed = data_observed
        self.degree = degree
        self.nc = nc

    # 係数を格納する行列の作成
    def create_polynomial(self):
        list = np.zeros((self.degree+1, 2))
        list[:, 1] = np.arange(self.degree+1)

        return list

    # 回帰係数の計算
    def calculate(self):
        self.poly_list = self.create_polynomial()
        x_observed = np.ones(
            (self.data_observed.shape[0], self.poly_list.shape[0]))
        y_observed = np.array(self.data_observed[:, 1])

        for i in range(1, self.poly_list.shape[0]):
            x_observed[:, i] = self.data_observed[:, 0] ** self.poly_list[i, 1]

        self.poly_list[:, 0] = np.linalg.inv(
            x_observed.T @ x_observed + self.nc * np.eye(x_observed.shape[1]))  @ x_observed.T @ y_observed.T

        return self.poly_list

    def plot(self):
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)

        x_ideal = np.arange(self.data_observed[:, 0].min(),
                            self.data_observed[:, 0].max(), 0.01)
        y_ideal = np.full(x_ideal.shape[0], self.poly_list[0, 0])
        fx_name = "$y={:.2f}".format(self.poly_list[0, 0])

        for i in range(1, self.poly_list.shape[0]):
            y_ideal += (self.poly_list[i, 0] *
                        x_ideal ** self.poly_list[i, 1])
            fx_name += "{:+.2f}x".format(self.poly_list[i, 0])
            if(self.poly_list[i, 1] == 1):
                continue
            fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 1])

        fx_name += "$"

        ax.scatter(
            self.data_observed[:, 0], self.data_observed[:, 1],
            label="observed value")
        print(fx_name)
        ax.plot(x_ideal, y_ideal, color="red", label=fx_name)
        ax.legend()

# 最小二乗法を実現するクラス(2変数)


class LeastSqueares3D(LeastSqueares):  # 1変数のクラスを継承

    # 係数を格納する行列の作成
    def create_polynomial(self):
        list = np.zeros((1, 3))

        for i in range(1, self.degree+1):
            for j in range(i+1):
                list = np.append(list, np.array([[0, i-j, j]]), axis=0)

        return list

    # 回帰係数の計算
    def calculate(self):
        self.poly_list = self.create_polynomial()
        xy_observed = np.ones(
            (self.data_observed.shape[0], self.poly_list.shape[0]))
        z_observed = np.array(self.data_observed[:, 2])

        for i in range(1, self.poly_list.shape[0]):
            xy_observed[:, i] = (self.data_observed[:, 0] ** self.poly_list[i, 1]
                                 * self.data_observed[:, 1] ** self.poly_list[i, 2])

        self.poly_list[:, 0] = np.linalg.inv(
            xy_observed.T @ xy_observed + self.nc * np.eye(xy_observed.shape[1]))  @ xy_observed.T @ z_observed.T

        return self.poly_list

    # 結果のプロット
    def plot(self):
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection="3d")

        x_ideal = np.arange(self.data_observed[:, 0].min(),
                            self.data_observed[:, 0].max(), 0.01)
        y_ideal = np.arange(self.data_observed[:, 1].min(),
                            self.data_observed[:, 1].max(), 0.01)
        z_ideal = np.full((x_ideal.shape[0], y_ideal.shape[0]),
                          self.poly_list[0, 0]).T
        x_ideal, y_ideal = np.meshgrid(x_ideal, y_ideal)
        fx_name = "$z={:.2f}".format(self.poly_list[0, 0])

        print(x_ideal.shape, y_ideal.shape, z_ideal.shape)

        for i in range(1, self.poly_list.shape[0]):
            z_ideal += (self.poly_list[i, 0] *
                        x_ideal ** self.poly_list[i, 1] *
                        y_ideal ** self.poly_list[i, 2])
            fx_name += "{:+.2f}".format(self.poly_list[i, 0])
            if(self.poly_list[i, 1] >= 1):
                fx_name += "x"
                if(self.poly_list[i, 1] > 1):
                    fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 1])
            if(self.poly_list[i, 2] >= 1):
                fx_name += "y"
                if(self.poly_list[i, 2] > 1):
                    fx_name += "^{{{:.0f}}}".format(self.poly_list[i, 2])

        fx_name += "$"

        ax.scatter(
            self.data_observed[:, 0], self.data_observed[:, 1],
            self.data_observed[:, 2], label="observed value")
        print(fx_name)
        ax.plot_wireframe(x_ideal, y_ideal, z_ideal,
                          color="red", label=fx_name)
        ax.legend()

# main


def main():
    csv = sys.argv[1]
    degree = int(sys.argv[2])
    nc = float(sys.argv[3])
    data_observed = np.loadtxt(
        fname=csv, dtype="float", delimiter=',', skiprows=1)  # csv の読み込み
    if(data_observed.shape[1] == 2):
        data = LeastSqueares(data_observed, degree, nc)
    elif(data_observed.shape[1] == 3):
        data = LeastSqueares3D(data_observed, degree, nc)
    data.calculate()
    data.plot()
    plt.savefig("ex03(" + csv + ").png")
    return


if __name__ == "__main__":
    main()
