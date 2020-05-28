from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CNormalizerDNN
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
LAYER = 'features:relu2'

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

    # Create LD
    feat_extr = CNormalizerDNN(dnn, out_layer=LAYER)
    clf = CClassifierMulticlassOVA(CClassifierSVM,
                                   kernel=CKernelRBF(),
                                   preprocess=feat_extr)

    # Multiprocessing
    clf.n_jobs = 16

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Xval
    xval_params = {'C': [1e-2, 1e-1, 1, 10, 100],
                   'kernel.gamma': [1e-3, 1e-2, 1e-1, 1]}

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
        metric='accuracy'
    )

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Dump to disk
    with open("dnr_best_params.log", "a") as f:
        f.write("LD[{0}] best params: {1}\n".format(LAYER, [(k, best_params[k]) for k in sorted(best_params)]))
