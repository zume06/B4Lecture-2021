import numpy as np


class PCA:
    def __init__(self, dim):
        self.dim = dim

        self.eigen_value = None
        self.eigen_vector = None
        self.contribution_rate = None

    def fit(self, X):
        err_msg = 'The dimensions are different, expected {}, but {} given'.format(
            X.shape[1], self.dim
        )
        assert X.shape[1] == self.dim, err_msg

        # data_length = X.shape[0]

        cov_matrics = np.cov(X, bias=True)
        self.eigen_value, self.eigen_vector = np.linalg.eig(cov_matrics)

        contribution_rate = np.zeros(self.dim)
        for i in range(self.dim):
            contribution_rate[i] = self.eigen_value[i] / np.sum(self.eigen_value)
        self.contribution_rate = contribution_rate

        return self

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass
