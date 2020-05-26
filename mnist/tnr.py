from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CNormalizerMinMax
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR

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

    # Create LD
    tsne = CReducerPTSNE(n_components=2,
                         n_hiddens=64,
                         epochs=100,
                         batch_size=128,
                         preprocess=None,
                         random_state=random_state)
    nmz = CNormalizerMinMax(preprocess=tsne)
    LD = CClassifierMulticlassOVA(classifier=CClassifierKDE,
                                   kernel=CKernelRBF(gamma=100),
                                   preprocess=nmz)

    # DEBUG
    tsne.verbose = 1

    # Create DNR
    layers = ['features:relu4', 'features:relu3']
    combiner = CClassifierMulticlassOVA(
        CClassifierSVM, kernel=CKernelRBF(gamma=1), C=1)
    layer_clf = LD
    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # # Setting layer classifiers parameters (avoid xval)
    # dnr.set_params({
    #     'features:relu2.C': 10,
    #     'features:relu2.kernel.gamma': 1e-3,
    #     'features:relu3.C': 1,
    #     'features:relu3.kernel.gamma': 1e-3,
    #     'features:relu4.C': 1e-1,
    #     'features:relu4.kernel.gamma': 1e-2,
    #     'clf.C': 1e-4,
    #     'clf.kernel.gamma': 1
    # })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 1     # DEBUG
    dnr.fit(tr_sample.X, tr_sample.Y)
    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('dnr')



