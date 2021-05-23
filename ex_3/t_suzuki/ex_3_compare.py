import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def reg_2d(x, y, d, l):
    """
    input
    x : array_like, shape(M,)
        x-coordinates of the M sample points (x[i], y[i])
    y : array_like, shape(M,)
        y-coordinates of the sample points
    d : int
        regression line degrees

    output
    x_label : regression x label
    y_reg   : regression y value
    w       : regression factor
    """

    phi = np.array([x ** n for n in range(d + 1)]).T
    I = np.eye(d+1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)+l*I), phi.T), y)
    
    x_label = np.arange(x.min(), x.max(), 0.1)
    y_reg = np.array([np.dot(w, np.array([np.sum(x ** n) for n in range(d + 1)])) for x in x_label])

    return x_label, y_reg, w

def get_label_2d(w):
    """
    input  : regression factor
    output : regression label
    """
    label = 'y='
    d = len(w)

    for i, w_i in enumerate(w[::-1]):
        if i == 0:
            if i == d-2:
                label += f'{w_i:.2}x'
            else:
                label += f'{w_i:.2}$x^{{{d-1-i}}}$'
        elif i == d-2:
            label += f'{w_i:+.2}x'
        elif i == d-1:
            label += f'{w_i:+.2}'
        else:
            label += f'{w_i:+.2}$x^{{{d-1-i}}}$'

    return label


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="input file path")
    parser.add_argument("--reg_x_d", type=int, required=True, help="regression degrees of x")
    parser.add_argument("--lamda", type=float, default=0.5, help="regularization parameters")
    parser.add_argument("--output_name", type=str, default='result', help='output file name')
    args = parser.parse_args()

    if not os.path.exists('./out'):
        os.makedirs('./out')
    
    # load data
    data = pd.read_csv(args.input_path)
    x, y = np.array(data.x1), np.array(data.x2)

    # regression
    x_reg_1, y_reg_1, w_1 = reg_2d(x, y, args.reg_x_d, 0)
    x_reg_5, y_reg_5, w_5 = reg_2d(x, y, args.reg_x_d, 0.5)
    x_reg_10, y_reg_10, w_10 = reg_2d(x, y, args.reg_x_d, 3.0)
    # get regression label
    reg_label_1 = get_label_2d(w_1)
    reg_label_5 = get_label_2d(w_5)
    reg_label_10 = get_label_2d(w_10)

    # setting graphic space 
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # plot data and add label
    ax.scatter(x, y, facecolor='None', edgecolors='blue', label='Observed data')
    ax.plot(x_reg_1, y_reg_1, color='orange', label='l=0.1 ' + reg_label_1)
    # ax.plot(x_reg_5, y_reg_5, color='orange', label='l=0.5 ' + reg_label_5)
    ax.plot(x_reg_10, y_reg_10, color='red', label='l=3.0 ' + reg_label_10)
 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(color='gray', linestyle='--')
    ax.legend(loc='upper right')
    
    # show and save fig
    plt.show(block=True)
    fig.savefig(f'./out/{args.output_name}')


if __name__ == "__main__":
    main()
