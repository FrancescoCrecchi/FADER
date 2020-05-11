from secml.array import CArray
from secml.ml.classifiers import CClassifier
from mycode.kdeclassifier import KDEClassifier


class CClassifierKDEBayes(CClassifier):

    def __init__(self, bandwidth=1.0, kernel='gaussian', preprocess=None):
        super().__init__(preprocess=preprocess)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._model = KDEClassifier(bandwidth=self.bandwidth, kernel=self.kernel)

    def _fit(self, dataset):
        # Unpack data
        X, y = dataset.X.tondarray(), dataset.Y.tondarray()
        # Fit KDEClassifier
        self._model.fit(X, y)

        return self

    def _forward(self, x):
        pred = self._model.predict_proba(x.tondarray())
        return CArray(pred)

    def _backward(self, w):
        """Compute the decision function gradient wrt x, and accumulate w."""
        # TODO: HANDLE `w is None` CASE!
        w = w.atleast_2d()
        X = self._cached_x.tondarray()
        grad = CArray.zeros(X.shape)
        # TODO: NEED TO VECTORIZE THIS!
        for i, x in enumerate(X):
            gi_ = self._model.grad_x(x, w[i, :].argmax())
            grad[i, :] = CArray(gi_)

        return grad


if __name__ == '__main__':
    from secml.data.loader import CDLRandomBlobs

    random_state = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters

    from secml.data.loader import CDLRandomBlobs

    dataset = CDLRandomBlobs(n_features=n_features,
                             centers=centers,
                             cluster_std=cluster_std,
                             n_samples=n_samples,
                             random_state=random_state).load()

    n_tr = 1000  # Number of training set samples
    n_ts = 250  # Number of test set samples

    # Split in training and test
    from secml.data.splitter import CTrainTestSplit

    splitter = CTrainTestSplit(
        train_size=n_tr, test_size=n_ts, random_state=random_state)
    tr, ts = splitter.split(dataset)

    # Normalize the data
    from secml.ml.features import CNormalizerMinMax

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    ts.X = nmz.transform(ts.X)

    # Training a classifier
    clf = CClassifierKDEBayes(bandwidth='auto')
    clf.fit(tr)

    # Compute predictions on a test set
    y_pred = clf.predict(ts.X)

    # Evaluate the accuracy of the classifier
    from secml.ml.peval.metrics import CMetricAccuracy
    acc = CMetricAccuracy().performance_score(y_true=ts.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))

    # Plot decision regions
    from secml.figure import CFigure

    fig = CFigure(width=5, height=5)

    # Convenience function for plotting the decision function of a classifier
    fig.sp.plot_decision_regions(clf, n_grid_points=200)

    fig.sp.plot_ds(ts)
    fig.sp.grid(grid_on=False)

    fig.sp.title("Classification regions")
    fig.sp.text(0.01, 0.01, "Accuracy on test set: {:.2%}".format(acc),
                bbox=dict(facecolor='white'))
    fig.savefig('kdebayes_blobs.png')

    # Test gradient
    x = tr.X[:10, :]
    w = clf.forward(x)
    grad = clf.gradient(x, w=w)
    print(grad.shape)

    # Numerical gradient check
    from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases
    CClassifierTestCases.setUpClass()
    CClassifierTestCases()._test_gradient_numerical(clf, ts.X[0, :])

    print("done?")

