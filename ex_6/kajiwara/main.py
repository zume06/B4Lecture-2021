import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pca

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    if args.result_path:
        result_path = Path(args.result_path)
        timestamp = datetime.now().strftime(TIME_TEMPLATE)
        result_path = result_path/timestamp
        if not result_path.exists():
            try:
                result_path.mkdir(parents=True)
            except Exception as err:
                print(err)

    # loading data
    df1 = pd.read_csv('../data1.csv', header=None)
    df2 = pd.read_csv('../data2.csv', header=None)
    df3 = pd.read_csv('../data3.csv', header=None)

    # df to nd ndarray
    data1 = df1.values
    data2 = df2.values
    data3 = df3.values

    # data1.csv
    pca1 = pca.PCA(dim=data1.shape[1])
    pca1.fit(data1)
    w_t = pca1.eigen_vector.T
    ax_x = np.arange(-2, 2, 0.1)
    ax_y = []
    for i, w in enumerate(w_t):
        ax_y.append(ax_x*w[0]/w[1])

    plt.axes().set_aspect('equal')
    for i, _ax_y in enumerate(ax_y):
        plt.plot(ax_x, _ax_y, label='contribution rate: {}'.format(pca1.contribution_rate[i]))
    plt.scatter(data1[:, 0], data1[:, 1], label='data1')
    plt.title('data1.csv')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(result_path/'data1.png')
    plt.clf()
    plt.close()

    # data2.csv
    pca2 = pca.PCA(dim=data2.shape[1])
    pca2.fit(data2)
    w_t = pca2.eigen_vector.T
    ax_x = np.arange(-2, 2, 0.1)
    ax_y = []
    ax_z = []
    for i, w in enumerate(w_t):
        ax_y.append(ax_x*w[1]/w[0])
        ax_z.append(ax_x*w[2]/w[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(w_t)):
        plt.plot(ax_x, ax_y[i], ax_z[i], label='contribution rate: {}'.format(pca2.contribution_rate[i]))
    ax.set_title("data2.csv")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2])
    plt.savefig(result_path/'data2,png')
    plt.clf()
    plt.close()

    data2_transformed = pca2.transform(data2)[:, :2]
    print(data2_transformed.shape)

    plt.axes().set_aspect('equal')
    plt.scatter(data2_transformed[:, 0], data2_transformed[:, 1], label='data2 transformed')
    plt.title('data2 transformed')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(result_path/'data2_transformed.png')
    plt.clf()
    plt.close()

    # data3.csv
    pca3 = pca.PCA(dim=data3.shape[1])
    pca3.fit(data3)
    print("pca3 contribution rate:", pca3.contribution_rate)


if __name__ == "__main__":
    description = 'Example: python main.py ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-rs', '--result-path', help='path to save the result')

    args = parser.parse_args()

    main(args)
