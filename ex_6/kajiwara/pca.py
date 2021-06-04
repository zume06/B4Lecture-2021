import numpy as np


class PCA:
    def __init__(self, dim):
        self.dim = dim

        self.eigen_value = None
        self.eigen_vector = None
        self.contribution_rate = None

    def fit(self, X):
        '''
        X: (dim, length)
        '''

        err_msg = 'The dimensions are different, expected {}, but {} given'.format(
            X.shape[1], self.dim
        )
        assert X.shape[1] == self.dim, err_msg

        cov_matrics = np.cov(X.T, bias=True)
        self.eigen_value, self.eigen_vector = np.linalg.eig(cov_matrics)
        self.eigen_vector = self.eigen_vector

        # idx = np.argsort(self.eigen_value)[::-1]
        # self.eigen_value = self.eigen_value[idx]
        # self.eigen_vector = self.eigen_vector[:, idx]

        contribution_rate = np.zeros(self.dim)
        for i in range(self.dim):
            contribution_rate[i] = self.eigen_value[i] / np.sum(self.eigen_value)
        self.contribution_rate = contribution_rate

        return self

    def transform(self, X):
        return np.dot(X, self.eigen_vector)

    def fit_transform(self, X):
        self = self.fit(X)
        return np.dot(X, self.eigen_vector)
