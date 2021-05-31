import numpy as np
import matplotlib.pyplot as plt
import sys


def minimax(data, c_size):
    data_size, n_features = data.shape
    init_codebook = np.zeros((c_size, n_features))
    rng = np.random.default_rng()
    cent = rng.choice(data)
    init_codebook[0] = cent
    d = np.linalg.norm(data - cent, axis=1)
    for i in range(1, c_size):
        cent = data[np.argmax(d)]
        init_codebook[i] = cent
        d += np.linalg.norm(data - cent, axis=1)
    return init_codebook


def kmeans(data, c_size, init_codebook):
    data_size, n_features = data.shape
    cluster = np.zeros(data_size)
    new_codebooks = np.zeros((c_size, n_features))
    codebooks = init_codebook
    while 1:
        for i in range(data_size):
            distances = np.linalg.norm(data[i] - codebooks, axis=1)
            cluster[i] = np.argsort(distances)[0]
        for k in range(c_size):
            new_codebooks[k] = data[cluster == k].mean(axis=0)
        if np.allclose(codebooks, new_codebooks) is True:
            break
        codebooks = new_codebooks
    return new_codebooks, cluster


def main():
    args = sys.argv
    data = np.loadtxt(fname=args[1], delimiter=",", skiprows=1)

    c_size = 4
    init_codebook = minimax(data, c_size)
    codebooks, cluster = kmeans(data, c_size, init_codebook)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(codebooks.shape[0]):
        plt.scatter(data[:, 0][cluster == i], data[:, 1][cluster == i])

    """
    ax = fig.add_subplot(111, projection="3d")
    for i in range(codebooks.shape[0]):
        ax.scatter3D(
            data[:, 0][cluster == i], data[:, 1][cluster == i], data[:, 2][cluster == i]
        )
    """

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_zlabel("z")
    plt.title(args[2])
    plt.savefig(args[2])
    plt.clf
    plt.close


if __name__ == "__main__":
    main()
