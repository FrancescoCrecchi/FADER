from secml.array import CArray
from secml.ml import CKernelRBF, CNormalizerMeanStd, CClassifierPyTorch, CClassifier
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.classifiers.sklearn.c_classifier_svm_m import CClassifierSVMM
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets


class CClassifierMean(CClassifier):

    def _fit(self, x, y):
        return self

    def _forward(self, x):
        res = CArray.zeros((x.shape[0], self.n_classes))
        # x.shape = [n_samples, n_layer * n_classes] -> [n_classes|n_classes|...|n_classes]
        l = int(x.shape[1] / self.n_classes)
        for i in range(self.n_classes):
            x_i = CArray.zeros((x.shape[0], l))
            for j in range(l):
                x_i[:, j] = x[:, (j * self.n_classes)+i]
            res[:, i] = x_i.mean(1)
        return res

    def _backward(self, w):

        # HACK: Trying avoiding zero grad.
        w[w == -1] = 0

        c, l = self._cached_x.shape
        grad = CArray.ones((self.n_classes, l)) * (1/l)
        return w.atleast_2d().dot(grad)


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    dnn.load_model('cnn_cifar10.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create DNR
    layers = ['features:23', 'features:26', 'features:29']
    layer_clf = CClassifierSVMM(kernel=CKernelRBF(gamma=1), C=1)
    nmz = CNormalizerMeanStd()
    combiner = CClassifierMean(preprocess=nmz)
    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    '''
    Setting classifiers parameters (A. Sotgiu)
    CIFAR		    C	    gamma
    -----------------------------
    combiner	    1e-4	1
    features:23	    10	    1e-3
    features:26	    1	    1e-3
    features:29	    1e-1	1e-2
    '''
    dnr.set_params({
        'features:23.C': 10,
        'features:23.kernel.gamma': 1e-3,
        'features:26.C': 1,
        'features:26.kernel.gamma': 1e-3,
        'features:29.C': 1e-1,
        'features:29.kernel.gamma': 1e-2,
        # 'clf.C': 1e-4,
        # 'clf.kernel.gamma': 1
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 2     # DEBUG
    dnr.fit(tr_sample.X, tr_sample.Y)
    dnr.verbose = 0

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('dnr_mean')

    # X, y = CArray.rand((100, 30)), CArray.zeros((100,))
    # clf = CClassifierMean()
    # clf.fit(X, y)
    # clf._classes = CArray.arange(10)
    # res = clf.forward(X)
    # print("done?")