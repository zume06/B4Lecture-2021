# include flake8, black

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os


def polyfit():
    pass


def poly1d():
    pass


def main():
    # get current working directory
    path = os.path.dirname(__file__)
    fname = "/data/data3.csv"
    save_fname = fname.replace("data", "result", 1).replace(".csv", "_result.png")
    fname = path + fname
    save_fname = path + save_fname

    # load data file and convert to ndarray
    data = pd.read_csv(fname).values

    # if data is 2 dimensional
    if data.shape[1] == 2:
        x = data[:, 0]  # load x1
        y = data[:, 1]  # load x2
        # plot_2d(x1, x2, deg1, regular)
        fig = plt.figure()
        ax = fig.add_subplot(111, title="Observed data", xlabel="X", ylabel="Y")
        ax.scatter(x, y, s=12, c="darkblue", label="observed")
        ax.grid(ls="--")
        ax.legend(loc="best", fontsize=10)
        # plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=1, fontsize=12)
        plt.savefig(save_fname)
        plt.show()

    # if data is 3 dimensional
    elif data.shape[1] == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # pLot_3d(x1, x2, x3, deg1, deg2, regular)
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 8),
            gridspec_kw={"width_ratios": [8, 7]},
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.suptitle("Observed data", fontsize=16)
        plt.rcParams["font.size"] = 12
        ax[0, 0].set(title="X - Y", xlabel="X", ylabel="Y")
        ax[0, 0].scatter(x, y, s=20, c="darkblue", label="observed")
        ax[0, 0].grid(ls="--")
        ax[0, 0].legend(loc="best", fontsize=10)
        # ax[0, 0].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)
        ax[1, 0].set(title="X - Y (color)", xlabel="X", ylabel="Y")
        img = ax[1, 0].scatter(x, y, s=20, c=z, cmap="rainbow", label="observed")
        fig.colorbar(img, ax=ax[1, 0], aspect=30, pad=0.01)
        ax[1, 0].grid(ls="--")
        ax[1, 0].legend(loc="best", fontsize=10)
        # ax[1, 0].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)
        ax[0, 1].set(title="Y - Z", xlabel="Y", ylabel="Z")
        ax[0, 1].scatter(y, z, s=20, c="darkblue", label="observed")
        ax[0, 1].grid(ls="--")
        ax[0, 1].legend(loc="best", fontsize=10)
        # ax[0, 1].legend(bbox_to_anchor=(0.99, 0.01), loc="lower right", borderaxespad=0, fontsize=10)
        ax[1, 1].set(title="Z - X", xlabel="Z", ylabel="X")
        ax[1, 1].scatter(z, x, s=20, c="darkblue", label="observed")
        ax[1, 1].grid(ls="--")
        ax[1, 1].legend(loc="best", fontsize=10)
        # ax[1, 1].legend(bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0, fontsize=10)

        ax_pos_0 = ax[0, 0].get_position()
        ax_pos_1 = ax[1, 0].get_position()
        ax[0, 0].set_position(
            [ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height]
        )
        # fig.tight_layout()
        plt.savefig(save_fname)
        plt.show()

        # X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter3D(x, y, z, s=20, c="darkblue")
        ax.set(title="data3_3D", xlabel="X", ylabel="Y", zlabel="Z")

        """
        def init():
            ax.scatter3D(x, y, z, s=20, c="darkblue")
            ax.set(title="3D", xlabel="X", ylabel="Y", zlabel="Z")
            return fig
        """

        def update(i):
            ax.view_init(elev=30.0, azim=3.6 * i)
            return fig

        # Animate
        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        ani.save(path + "/result/data3_result3D.gif", writer="pillow")
        # ani.save(path + "/result/data3_result3D.mp4", writer="ffmpeg", dpi=100)
        plt.show()


if __name__ == "__main__":
    """
    # process args
    parser = argparse.ArgumentParser(
        description="apply filter (HPF, LPF, BFF, BEF) to wav file"
    )
    parser.add_argument(
        "sc",
        type=str,
        default="/wav/sample.wav",
        help="input filename with extension (Default : wav/sample.wav)",
    )
    parser.add_argument(
        "dst",
        type=str,
        default="/result/sample_filtered.wav",
        help="output filename with extension (Default : result/sample_filtered.wav)",
    )
    parser.add_argument(
        "--bef",
        type=int,
        nargs=2,
        metavar="freq",
        help="low and high frequency [Hz]",
    )

    args = parser.parse_args()
    """

    # main(args)
    main()
