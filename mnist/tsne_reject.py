from secml.array import CArray
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax
from secml.ml.kernels import CKernelRBF

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

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

    from secml.ml.peval.metrics import CMetric

    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Create layer_classifier
    feat_extr = CNormalizerDNN(dnn, out_layer='features:relu4')
    # Compose classifier
    tsne = CReducerPTSNE(n_components=2,
                         n_hiddens=64,
                         epochs=100,
                         batch_size=128,
                         preprocess=feat_extr,
                         random_state=random_state)
    nmz = CNormalizerMinMax(preprocess=tsne)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE,
                                   kernel=CKernelRBF(gamma=100),
                                   preprocess=nmz)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # DEBUG
    tsne.verbose = 1

    # # Xval
    # xval_params = {'C': [1e-1, 1, 10, 100],
    #                'kernel.gamma': [0.1, 1, 10, 100]}
    #
    # # Let's create a 3-Fold data splitter
    # from secml.data.splitter import CDataSplitterKFold
    #
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

    # We can now create a classifier with reject
    clf.preprocess = None  # TODO: "preprocess should be passed to outer classifier..."
    clf_rej = CClassifierRejectThreshold(clf, 0., preprocess=nmz)
    # We can now fit the clf_rej
    clf_rej.fit(tr_sample.X, tr_sample.Y)
    # Set threshold (FPR: 10%)
    # TODO: "..and set the rejection threshold for (D)NR to reject 10% of the samples when no attack is performed
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = clf_rej.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Dump to disk
    clf_rej.save('tsne_rej')
