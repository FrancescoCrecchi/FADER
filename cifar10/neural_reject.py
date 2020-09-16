from secml.ml.features import CNormalizerDNN, CNormalizerMinMax, CNormalizerMeanStd
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets

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

    # Create layer_classifier
    feat_extr = CNormalizerDNN(dnn, out_layer='features:29')
    clf = CClassifierSVM(kernel=CKernelRBF(), preprocess=feat_extr)
    # clf.n_jobs = 10

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # # Xval
    # xval_params = {'C': [1e-1, 1, 10, 100],
    #                'kernel.gamma': [0.1, 1, 10, 100]}
    #
    # # Let's create a 3-Fold data splitter
    # from secml.data.splitter import CDataSplitterKFold
    # xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)
    #
    # # Select and set the best training parameters for the classifier
    # clf.verbose = 1
    # print("Estimating the best training parameters...")
    # best_params = clf.estimate_parameters(
    #     dataset=tr_sample,
    #     parameters=xval_params,
    #     splitter=xval_splitter,
    #     metric='accuracy',
    #     perf_evaluator='xval'
    # )
    #
    # print("The best training parameters are: ",
    #       [(k, best_params[k]) for k in sorted(best_params)])


    # HACK: Avoid xval! (A. Sotgiu)
    clf.set_params({
        'C': 1e-1,
        'kernel.gamma': 1e-2
    })

    # We can now fit the clf_rej
    clf.verbose = 2 # DEBUG
    clf.fit(tr_sample.X, tr_sample.Y)
    clf.verbose = 0  # DEBUG END

    # Check test performance
    y_pred = clf.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("NR Accuracy: {}".format(acc))

    # We can now create a classifier with reject
    clf_rej = CClassifierRejectThreshold(clf, 0.)

    # Set threshold (FPR: 10%)
    # TODO: "..and set the rejection threshold for (D)NR to reject 10% of the samples when no attack is performed
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)
    # Dump to disk
    clf_rej.save('nr')