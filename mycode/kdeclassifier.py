import numpy as np
from mycode.kde import KDE
from sklearn.base import BaseEstimator, ClassifierMixin


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KDE(bw=self.bandwidth, 
                            kernel=self.kernel).fit(Xi) for Xi in training_sets]

        self.logpriors_ = np.array([np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets])
        return self
        
    def predict_proba(self, X):
        logprobs = np.vstack([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result #/ result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def grad_x(self, X, y=None):
        du_dx = np.vstack([model.grad_x(X) for model in self.models_]).T
        dv_du = 1.
        dy_dv = self.predict_proba(X)
        grad = dy_dv * dv_du * du_dx

        return grad[:, y][:, None].T if y is not None else grad

