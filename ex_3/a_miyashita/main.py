import argparse

import numpy as np
from matplotlib import pyplot as plt

def generalized_inv(x):
    return np.linalg.inv(x.T @ x) @ x.T

class LinearModel:

    def __init__(self, base_functions):
        self.base_functions = base_functions
        self.w = np.zeros(len(base_functions))

    def fit(self, x, y):
        phi = np.zeros((x.shape[0], self.w.shape[0]))

        for i in range(phi.shape[1]):
            phi[:, i] = self.base_functions[i](x)

        self.w =  generalized_inv(phi) @ y

    def predict(self, x):
        phi = np.zeros((x.shape[0], self.w.shape[0]))

        for i in range(phi.shape[1]):
            phi[:, i] = self.base_functions[i](x)

        return phi @ self.w


class polynomial:

    def __init__(self, dim_list):
        self.dim_list = dim_list

    def __call__(self, x):
        if x.ndim == 1:
            return x ** self.dim_list[0]

        else:
            result = 1.0
            for i in range(len(self.dim_list)):
                result *= x[:, i] ** self.dim_list[i]

            return result


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Linear model fitting")
    parser.add_argument("fname", type=str, help="input filename with extension .csv")
    parser.add_argument(
        "dim", type=int, help="dimension of polynomial used as base functions"
    )
    parser.add_argument("lambda", type=float, help="regularizer coefficient")
    args = parser.parse_args()

    data = np.loadtxt(args.fname, delimiter=',', skiprows=1)

    x = data[: ,: -1]
    y = data[:, -1]
    print(x.shape)
    print(y.shape)

    base_functions = []
    for i in range((args.dim + 1) ** x.shape[1]):
        j = i
        dim_list = [0] * x.shape[1]
        for k in range(x.shape[1]):
            dim_list[k] = j % (args.dim + 1)
            j = j // (args.dim + 1)
        base_functions.append(polynomial(dim_list))

    model = LinearModel(base_functions)

    model.fit(x, y)

    x1 = np.linspace(-6, 6)
    prediction = model.predict(x1)
    plt.plot(x1, prediction)
    plt.scatter(x[:, 0], y)
    plt.title("data1")
    plt.show()

    """
    x1 = np.linspace(-5, 5)
    x2 = np.linspace(0, 10)

    X1, X2 = np.meshgrid(x1, x2)
    X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)

    prediction = model.predict(X)
    prediction = prediction.reshape(50, 50)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, prediction, cmap='rainbow', linewidth=0)
    fig.colorbar(surf)
    ax.set_title("data3")

    ax.scatter3D(x[:, 0], x[:, 1], y)
    ax.view_init(elev=10, azim=10)
    plt.show()
    """