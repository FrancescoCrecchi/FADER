import os
import numpy as np
from mycode.kernels import Gaussian
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class KDE(object):

    def __init__(self, kernel='gaussian', bw='auto'):
        self._bw = bw
        # Set kernel
        if kernel == 'gaussian':
            self._kernel = kernel
        else:
            raise ValueError("Unsupported kernel!")
        
        # Room for data (instance-based classifier)
        self.data = None
        
        # To fit?
        self.to_fit = True
    
    def fit(self, X):
        # Simply store the data
        self.data = X

        # Set bandwidth
        if self._bw == 'auto':
            self._set_bandwidth()
        elif type(self._bw) is float:
            self.bw = self._bw
        else:
            raise ValueError("Bandwidth not recognized!")

        # Set kernel: for the 'Gaussian' kernel h = sigma
        gamma = 1/(2 * self.bw**2)
        self.kernel = Gaussian(gamma)

        # Mark as fit
        self.to_fit = False

        return self

    def _score_samples(self, X):
        
        # Compute density estimation
        res = np.sum(self.kernel(self.data, X), axis=0, keepdims=True)
        # Average on number of samples
        res /= self.data.shape[0]

        return res

    def score_samples(self, X):
        scores = self._score_samples(X)
        return scores
        
    def grad_x(self, X):
        
        # Compute overall gradient
        grad = np.sum(self.kernel.grad_x(self.data, X), axis=0, keepdims=True)
        # Average over dataset size
        grad /= self.data.shape[0]

        return grad

    def _set_bandwidth(self):
        # Auto-tune bandwidth using cross-validation
        bandwidths = 10 ** np.linspace(-2, 1, 1000)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {
            'bandwidth': bandwidths}, cv=3, verbose=1)
        grid.fit(self.data)
        self.bw = grid.best_params_['bandwidth']

