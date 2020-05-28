from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CNormalizerMinMax
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


LOGFILE = 'tnr_best_params.log'


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

    # Create LD
    tsne = CReducerPTSNE(epochs=250, batch_size=128, preprocess=None, random_state=random_state)
    nmz = CNormalizerMinMax(preprocess=tsne)
    LD = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=nmz, n_jobs=10)

    # Create DNR
    layers = ['features:relu4', 'features:relu3', 'features:relu2']
    combiner = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    layer_clf = LD
    tnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # Setting layer classifiers parameters (separate xval)
    tnr.set_params({
        'features:relu2.preprocess.preprocess.n_hiddens': [64, 64],
        'features:relu2.kernel.gamma': 100,
        'features:relu3.preprocess.preprocess.n_hiddens': [256, 256],
        'features:relu3.kernel.gamma': 100,
        'features:relu4.preprocess.preprocess.n_hiddens': [128, 128],
        'features:relu4.kernel.gamma': 100
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Combiner Xval
    xval_params = {
        'C': [1e-2, 1e-1, 1, 10, 100],
        'kernel.gamma': [1e-3, 1e-2, 1e-1, 1]    # No nmz!
    }

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Paralellize?
    combiner.n_jobs = 16

    # Select and set the best training parameters for the classifier
    combiner.verbose = 1    # DEBUG
    print("Estimating the best training parameters...")
    best_params = combiner.estimate_parameters(
        dataset=tr_sample,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy'
    )
    combiner.verbose = 0    # END DEBUG

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Dump to disk
    with open(LOGFILE, "a") as f:
        f.write("COMBINER best params: {:} \n".format([(k, best_params[k]) for k in sorted(best_params)]))

    # Fit DNR
    tnr.verbose = 1     # DEBUG
    tnr.fit(tr_sample.X, tr_sample.Y)
    # Set threshold (FPR: 10%)
    tnr.threshold = tnr.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = tnr.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Dump to disk
    tnr.save('dnr')

