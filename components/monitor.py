import time

from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml import CClassifier, CClassifierSVM, CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from toy.gamma_estimation import gamma_estimation


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        clf = args[0]
        if clf.logtime_data is not None:
            clf._logtime_data.append((te - ts) * 1000)
            if clf.verbose:
                clf.logger.info('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class CClassifierMonitored(CClassifier):

    def __init__(self, clf):
        super(CClassifierMonitored, self).__init__()
        self._clf = clf
        # Internals
        self._caching = None
        self._logtime_data = []

    def _fit(self, x, y):
        raise ValueError("This should never be called! (Monitoring a pre-trained classifier)")

    def forward(self, x, caching=True):
        self._caching = caching
        return super().forward(x, self._caching)

    @timeit
    def _forward(self, x):
        return self._clf.forward(x, caching=self._caching)

    def _backward(self, w):
        return self._clf.gradient(self._cached_x, w)

    @property
    def classes(self):
        return self._clf.classes

    @property
    def n_classes(self):
        return self._clf.n_classes

    @property
    def n_features(self):
        return self._clf.n_features

    @property
    def logtime_data(self):
        return CArray(self._logtime_data)


if __name__ == '__main__':
    seed = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, -2], [-2, 2], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=seed).load()

    tr, ts = CTrainTestSplit(test_size=0.3, random_state=seed).split(ds)

    # Create a SVM-RBF classifier
    gamma = gamma_estimation(tr, factor=0.5)
    clf = CClassifierSVM(kernel=CKernelRBF(gamma=gamma))

    # Fit
    clf.verbose = 2
    clf.fit(tr.X, tr.Y)
    clf.verbose = 0

    # Wrapping for measuring predict timing
    clf = CClassifierMonitored(clf)

    # Test performance
    clf.verbose = 1
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Test set accuracy: {:.2f}".format(acc))

    print(clf.logtime_data)

    # # Test gradient
    # x = ts.X[0, :]
    # w = CArray.zeros(len(centers), )
    # w[1] = 1.
    # grad = clf.gradient(x, w)
    # print(grad)
    #
    # # Numerical gradient check
    # from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases
    #
    # CClassifierTestCases.setUpClass()
    # CClassifierTestCases()._test_gradient_numerical(clf, ts.X[-1, :])
