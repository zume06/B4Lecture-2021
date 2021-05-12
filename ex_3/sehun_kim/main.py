import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import csv



def csv_open(fname):
    file = pd.read_csv(fname)
    file = np.array(file)
    return file


def reg_2d(data, degree):
    x, y = data.T
    phi = np.zeros([len(x), degree+1])
    for i in range(degree+1):
        phi[:, i] += x ** i

    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), y)

    reg_x = np.arange(min(x), max(x), 0.01)
    reg_y = np.zeros(len(reg_x))

    for i in range(len(reg_x)):
        for j in range(len(w)):
            reg_y[i] += (reg_x[i] ** j) * w[j]

    plt.scatter(x, y)
    plt.plot(reg_x, reg_y, label='Regression', color='red')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("result_2d.png")


def reg_3d(data, degree):
    x, y, z = data.T
    phi = np.zeros([len(x), degree * 2 + 1])
    for i in range(degree+1):
        phi[:, i] += x ** i
    for j in range(degree):
        phi[:, j + degree + 1] += y ** (j +1)

    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), z)

    reg_x = np.arange(min(x), max(x), 0.01)
    reg_y = np.arange(min(y), max(y), (max(y)-min(y)) / ((max(x)-min(x)) /0.01))
    reg_x, reg_y = np.meshgrid(reg_x, reg_y)
    reg_z = np.zeros([len(reg_y), len(reg_x)])
    for i in range(0, degree + 1):
        reg_z += (reg_x ** i) * w[i]
    for j in range(1, degree + 1):
        reg_z += (reg_y ** j) * w[j + degree]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')
    ax.scatter(x, y, z, label = 'Original data')
    ax.plot_wireframe(reg_x, reg_y, reg_z , color = 'g')

    plt.savefig("result_3d.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Program for plotting regression equation.\nFile name, degree for regression equation are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-d", dest="degree", type=int,
                        help='Degree for regression equation', required=True)
    args = parser.parse_args()
    data = csv_open(args.filename)
    dimension = np.shape(data)[1]
    if dimension == 3:
        reg_3d(data, args.degree)
    else:
        reg_2d(data, args.degree)

    print("Program terminated")
