import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegression:
    def __init__(self, digree=1):
        self.digree = digree

    def fit(self, X, y, sample_weight=None):
        '''
        Fit linear model.
        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values
        sample_weight : array, default=None
            Individual weights for each sample
        Returns
        -------
        self : returns an instance of self.
        '''

        poly = PolynomialFeatures(self.digree)
        phi = poly.fit_transform(X)
        # phi = np.zeros((len(X), self.digree+1))
        # for i in range(self.digree+1):
        #     p = X**i
        #     phi[:, i] = p.ravel()

        A = np.dot(phi.T, phi)
        B = np.dot(phi.T, y)
        # self.coef = np.dot(np.linalg.inv(A), B)
        self.coef = np.dot(np.linalg.pinv(A), B)

        return self
