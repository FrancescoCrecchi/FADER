from secml.array import CArray
from secml.ml import CKernelRBF, CClassifierSVM
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create DNR
    layers = ['features:relu4', 'features:relu3', 'features:relu2']
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    '''
    Setting classifiers parameters (A. Sotgiu)
    MNIST		    C	    gamma
    -----------------------------
    combiner	    1e-1	1
    features:relu2	10	    1e-3
    features:relu3	10	    1e-2
    features:relu4	1	    1e-2
    '''
    dnr.set_params({
        'features:relu2.C': 10,
        'features:relu2.kernel.gamma': 1e-3,
        'features:relu3.C': 10,
        'features:relu3.kernel.gamma': 1e-2,
        'features:relu4.C': 1,
        'features:relu4.kernel.gamma': 1e-2,
        'clf.C': 1e-1,
        'clf.kernel.gamma': 1
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 2     # DEBUG
    dnr.fit(tr_sample.X, tr_sample.Y)

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('dnr')



