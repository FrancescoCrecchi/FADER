from secml.ml import CNormalizerDNN, CNormalizerMinMax, CClassifierSVM, CKernelRBF
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.array import CArray

from mnist import get_datasets, mnist


if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = mnist()
    dnn.load_model('mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric

    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Create detector
    feat_extr = CNormalizerDNN(dnn, out_layer='fc2')
    nmz = CNormalizerMinMax(preprocess=feat_extr)
    clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    clf_rej = CClassifierRejectThreshold(clf, 0., preprocess=nmz)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=10000, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=1000, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Xval
    xval_params = {'C': [1e-3, 1, 1000],
                   'kernel.gamma': [0.01, 1, 10]}

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Select and set the best training parameters for the classifier
    clf.verbose = 1
    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=tr_sample,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # We can now fit the classifier
    clf_rej.fit(tr_sample.X, tr_sample.Y)
    clf_rej.save('clf_rej.pkl')