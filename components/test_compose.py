from secml.array import CArray
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


def eval(clf, dset):
    X, y = dset.X, dset.Y
    # Predict
    y_pred = clf.predict(X)
    # Evaluate the accuracy of the classifier
    return CMetricAccuracy().performance_score(y, y_pred)

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    tr, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('../mnist/cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric

    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Wrap it with `CNormalizerDNN`
    dnn_feats = CNormalizerDNN(dnn, out_layer='features:relu4')

    # Compose classifier
    feat_extr = CReducerPTSNE(preprocess=dnn_feats, random_state=random_state)
    nmz = CNormalizerMinMax(preprocess=feat_extr)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=nmz)

    # DEBUG
    feat_extr.verbose = 1

    # # Xval
    # xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)
    #
    # def compute_hiddens(n_hiddens, n_layers):
    #     return sum([[[l] * k for l in n_hiddens] for k in range(1, n_layers+1)], [])
    #
    # params_grid = {
    #     'preprocess.preprocess.n_hiddens': [128, 64], #compute_hiddens([8, 64, 128], 1),
    #     # 'preprocess.preprocess.n_components': [2, 4]
    #     'kernel.gamma': [1, 10, 100]
    # }
    # clf.verbose = 1
    # best_params = clf.estimate_parameters(clf_tr,
    #                                       parameters=params_grid,
    #                                       splitter=xval_splitter,
    #                                       metric='accuracy')
    # print("The best training parameters are: ",
    #       [(k, best_params[k]) for k in sorted(best_params)])

    # HACK: Setting "best params" by hand!
    clf.set_params({
        'preprocess.preprocess.n_hiddens': 128,
        'kernel.gamma': 100
    })

    # DEBUG
    clf.n_jobs = 10

    # Fit
    clf.fit(tr_sample.X, tr_sample.Y)
    tr_acc = eval(clf, tr_sample)
    print("Accuracy on training set: {:.2%}".format(tr_acc))

    # Test
    ts_acc = eval(clf, ts_sample)
    print("Accuracy on test set: {:.2%}".format(ts_acc))

    # # --------- Plot ---------
    # embds = clf.preprocess.transform(clf_tr.X)
    # clf.preprocess = None
    #
    # fig = CFigure(10, 12)
    # # Decision function
    # fig.sp.plot_decision_regions(clf, n_grid_points=200)
    # # Plot embds dataset
    # foo_ds = CDataset(embds, clf_tr.Y)
    # fig.sp.plot_ds(foo_ds, alpha=0.5)
    # # Extras
    # fig.sp.legend()
    # fig.sp.grid()
    # fig.savefig('test_compose.png')
    #
    # # Restore preprocessing
    # clf.preprocess = nmz

    # --------- Backward ---------

    # Test gradient
    x = tr_sample.X[0, :]
    w = clf.forward(x)
    grad = clf.gradient(x, w=w)
    print(grad.shape)

    # Numerical gradient check
    from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases

    CClassifierTestCases.setUpClass()
    CClassifierTestCases()._test_gradient_numerical(clf, x)

    print("done?")
