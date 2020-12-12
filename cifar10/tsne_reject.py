from secml.array import CArray
from secml.data.splitter import CDataSplitterKFold
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax, CNormalizerMeanStd
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

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
    # Compose classifier
    tsne = CReducerPTSNE(epochs=1000,
                         batch_size=32,
                         preprocess=feat_extr,
                         random_state=random_state)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=tsne)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # # Xval
    # def compute_hiddens(n_hiddens, n_layers):
    #     return sum([[[l] * k for l in n_hiddens] for k in range(1, n_layers+1)], [])
    #
    # xval_params = {
    #     'preprocess.preprocess.n_hiddens': compute_hiddens([64, 256], 2),
    #     'kernel.gamma': [1e-3, 1, 100, 1000]
    # }
    #
    # # Let's create a 3-Fold data splitter
    # xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)
    #
    # # Parallel?
    # clf.n_jobs = 8
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
    # clf.verbose = 0
    #
    # print("The best training parameters are: ",
    #       [(k, best_params[k]) for k in sorted(best_params)])

    # Setting best params (external xval)
    clf.set_params({
        'kernel.gamma': 10,
        'preprocess.n_hiddens': [256, 256]}
    )
    # Expected performance: 0.9735

    # We can now create a classifier with reject
    clf.preprocess = None  # TODO: "preprocess should be passed to outer classifier..."
    clf_rej = CClassifierRejectThreshold(clf, -1000., preprocess=tsne)

    # We can now fit the clf_rej
    clf_rej.preprocess.verbose = 1  # DEBUG
    clf_rej.fit(tr_sample.X, tr_sample.Y)

    # Set threshold (FPR: 10%)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = clf_rej.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric, CMetricAccuracy

    acc = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Dump to disk
    clf_rej.save('tsne_rej')
