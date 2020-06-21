from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CNormalizerMinMax, CNormalizerMeanStd
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets


LOGFILE = 'tnr_best_params.log'


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    dnn.load_model('cnn_cifar10.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create LD
    tsne = CReducerPTSNE(epochs=250, batch_size=128, preprocess=None, random_state=random_state)
    nmz = CNormalizerMinMax(preprocess=tsne)
    LD = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=nmz, n_jobs=10)

    # Create DNR
    layers = ['features:23', 'features:26', 'features:29']
    combiner = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    layer_clf = LD
    tnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # Setting layer classifiers parameters (separate xval)
    tnr.set_params({
        'features:23.preprocess.preprocess.n_hiddens': [256],
        'features:23.kernel.gamma': 100,
        'features:26.preprocess.preprocess.n_hiddens': [256, 256],
        'features:26.kernel.gamma': 1000,
        'features:29.preprocess.preprocess.n_hiddens': [256, 256],
        'features:29.kernel.gamma': 1000,
        'clf.C': 100,
        'clf.kernel.gamma': 1
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    tnr.fit(tr_sample.X, tr_sample.Y)
    # Set threshold (FPR: 10%)
    tnr.threshold = tnr.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = tnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Dump to disk
    tnr.save('tnr')

