from secml.array import CArray
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

GAMMA = [0.01, 0.1, 1, 10, 100, 1000, 1E4, 1E5]

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
    acc = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Create layer_classifier
    feat_extr = CNormalizerDNN(dnn, out_layer='features:relu4')
    # Compose classifier
    tsne = CReducerPTSNE(epochs=250, batch_size=128, random_state=random_state, preprocess=feat_extr)
    nmz = CNormalizerMinMax(preprocess=tsne)

    # MAIN LOOP HERE!
    for gamma in GAMMA:
        # 1. Create clf with desired gamma
        preproc = nmz.deepcopy()
        clf = CClassifierMulticlassOVA(classifier=CClassifierKDE,
                                       kernel=CKernelRBF(gamma),
                                       preprocess=preproc,
                                       n_jobs=16)
        # 2. Fit
        clf.fit(tr_sample.X, tr_sample.Y)
        # 3. Wrap with reject
        clf.preprocess = None  # TODO: "preprocess should be passed to outer classifier..."
        clf_rej = CClassifierRejectThreshold(clf, 0., preprocess=preproc)

        # Set threshold (FPR: 10%)
        # TODO: "..and set the rejection threshold for (D)NR to reject 10% of the samples when no attack is performed
        clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

        # Check test performance
        y_pred = clf_rej.predict(ts.X, return_decision_function=False)
        acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
        print("- clf_rej.gamma: {0} -> Accuracy: {1:.2f}".format(gamma, acc))

        # 4. Dump to disk
        clf_rej.save('tsne_rej_test_gamma_{:}'.format(gamma))