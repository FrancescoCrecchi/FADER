import os

from secml.data import CDataset

from mnist import mnist
from ptSNE import ptSNE, scatter_plot
from c_classifier_kdebayes import CClassifierKDEBayes
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.figure import CFigure


def eval(clf, dset):
    X, y = dset.X, dset.Y
    # Predict
    y_pred = clf.predict(X)
    # Evaluate the accuracy of the classifier
    return CMetricAccuracy().performance_score(y, y_pred)


N_TRAIN_DNN = 30000
N_TRAIN_CLF = 3000

if __name__ == '__main__':
    import setGPU
    random_state = 999

    # Prepare data
    from secml.data.loader import CDataLoaderMNIST
    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    ts = loader.load('testing')
    # Normalize the data
    tr.X /= 255
    ts.X /= 255

    # Get dnn
    dnn = mnist()
    if not os.path.exists("mnist.pkl"):
        dnn.verbose = 1
        dnn.fit(tr.X[:N_TRAIN_DNN, :])
        dnn.save_model("mnist.pkl")
    else:
        dnn.load_model("mnist.pkl")

    # Wrap it with `CNormalizerDNN`
    dnn_feats = CNormalizerDNN(dnn, out_layer='fc2')

    # Compose classifier
    sample = tr[N_TRAIN_DNN:N_TRAIN_DNN+N_TRAIN_CLF, :]
    feat_extr = ptSNE(sample,
                      d=2,
                      epochs=1000,
                      preprocess=dnn_feats,
                      random_state=random_state,
                      verbose=1)
    nmz = CNormalizerMinMax(preprocess=feat_extr)
    clf = CClassifierKDEBayes(bandwidth='auto',
                              preprocess=nmz)

    # Fit
    clf_tr = tr[N_TRAIN_DNN+N_TRAIN_CLF:N_TRAIN_DNN+2*N_TRAIN_CLF, :]
    clf.fit(clf_tr)

    tr_acc = eval(clf, clf_tr)
    print("Accuracy on training set: {:.2%}".format(tr_acc))

    # Test
    cl_ts = ts[:1000, :]
    ts_acc = eval(clf, cl_ts)
    print("Accuracy on test set: {:.2%}".format(ts_acc))

    # --------- Plot ---------
    embds = clf.preprocess.transform(clf_tr.X)
    clf.preprocess = None

    fig = CFigure(10, 12)
    # Decision function
    fig.sp.plot_decision_regions(clf, n_grid_points=200)
    # Plot embds dataset
    foo_ds = CDataset(embds, clf_tr.Y)
    fig.sp.plot_ds(foo_ds, alpha=0.5)
    # Extras
    fig.sp.legend()
    fig.sp.grid()
    fig.savefig('test_compose.png')

    # # --------- Backward ---------
    #
    # # Test gradient
    # x = sample.X[:10, :]
    # w = clf.forward(x)
    # grad = clf.gradient(x, w=w)
    # print(grad.shape)
    #
    # # Numerical gradient check
    # from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases
    #
    # CClassifierTestCases.setUpClass()
    # CClassifierTestCases()._test_gradient_numerical(clf, sample.X[0, :])

    print("done?")


