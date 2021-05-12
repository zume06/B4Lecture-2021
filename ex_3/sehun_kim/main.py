import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import argparse
import csv


def csv_open(fname):
    file = pd.read_csv(fname)
    file = np.array(file)
    return file


def reg_2d(data, degree):
    x, y = data.T
    X = np.zeros([len(x), degree+1])
    for i in range(degree+1):
        X[:, i] += x ** i

    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    reg_x = np.arange(min(x), max(x), 0.01)
    reg_y = np.zeros(len(reg_x))

    for i in range(len(reg_x)):
        for j in range(len(w)):
            reg_y[i] += (reg_x[i] ** j) * w[j]

    plt.scatter(x, y)
    plt.plot(reg_x, reg_y, label='Regression', color='red')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("result.png")


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
        x, y, z = data.T
        reg_3d(data, args.degree)
    else:
        
        reg_2d(data, args.degree)
    

    print("Program terminated")
