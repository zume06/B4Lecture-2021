import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def reg_ndim(x, y, d):
    """
    input
    x : array_like, shape(M,)
        x-coordinates of the M sample points (x[i], y[i])
    y : array_like, shape(M,)
        y-coordinates of the sample points
    d : int
        regression function dimension

    output
    x_label : regression x label
    y_reg   : regression y value
    w       : regression constant term
    """
    phi = np.array([x ** n for n in range(d + 1)]).T
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), y)
    
    x_label = np.arange(x.min(), x.max(), 0.1)
    y_reg = np.array([np.dot(w, np.array([np.sum(x ** n) for n in range(d + 1)])) for x in x_label])

    return x_label, y_reg, w


def get_reg_fomula(w):
    """
    input  : regression constant term
    output : regression function fomula
    """
    fomula = 'y='
    d = len(w)

    for i, w_i in enumerate(w[::-1]):
        if i == 0:
            if i == d-2:
                fomula += f'{w_i:.2}x'
            else:
                fomula += f'{w_i:.2}$x^{{{d-1-i}}}$'
        elif i == d-2:
            fomula += f'{w_i:+.2}x'
        elif i == d-1:
            fomula += f'{w_i:+.2}'
        else:
            fomula += f'{w_i:+.2}$x^{{{d-1-i}}}$'

    return fomula


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="input file path")
    parser.add_argument("--reg_dim", type=int, required=True, help="regression dimension")
    parser.add_argument("--output_name", type=str, default='result', help='output file name')
    args = parser.parse_args()

    if not os.path.exists('./out'):
        os.makedirs('./out')
    
    # load data
    data = pd.read_csv(args.input_path)
    x, y = np.array(data.x1), np.array(data.x2)

    # regression
    x_reg, y_reg, w = reg_ndim(x, y, args.reg_dim)
    # get regression fomula
    reg_fomula = get_reg_fomula(w)

    # setting graphic space 
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # plot data and add label
    ax.scatter(x, y, facecolor='None', edgecolors='blue', label='Observed data')
    ax.plot(x_reg, y_reg, color='red', label=reg_fomula)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(color='gray', linestyle='--')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02,), borderaxespad=0)
    
    # show and save fig
    plt.show(block=True)
    fig.savefig(f'./out/{args.output_name}')


if __name__ == "__main__":
    main()
