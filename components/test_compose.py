from setGPU import setGPU
setGPU(3)

import os

from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.features import CNormalizerDNN, CNormalizerMinMax
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE
from mnist.mnist import mnist


def eval(clf, dset):
    X, y = dset.X, dset.Y
    # Predict
    y_pred = clf.predict(X)
    # Evaluate the accuracy of the classifier
    return CMetricAccuracy().performance_score(y, y_pred)


N_TRAIN_DNN = 30000
N_TRAIN_CLF = 3000

if __name__ == '__main__':

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
    if not os.path.exists("../mnist/mnist.pkl"):
        dnn.verbose = 1
        dnn.fit(tr[:N_TRAIN_DNN, :])
        dnn.save_model("mnist.pkl")
    else:
        dnn.load_model("mnist.pkl")

    # Wrap it with `CNormalizerDNN`
    dnn_feats = CNormalizerDNN(dnn)#, out_layer='fc1')

    # Compose classifier
    sample = tr[N_TRAIN_DNN:N_TRAIN_DNN + N_TRAIN_CLF, :]
    feat_extr = CReducerPTSNE(n_components=2,
                              n_hiddens=64,
                              epochs=100,
                              batch_size=128,
                              preprocess=dnn_feats,
                              random_state=random_state,
                              verbose=1)
    nmz = CNormalizerMinMax(preprocess=feat_extr)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE,
                                   kernel=CKernelRBF(gamma=100),
                                   preprocess=nmz)

    # Sample for clf training
    clf_tr = tr[N_TRAIN_DNN + N_TRAIN_CLF:N_TRAIN_DNN + 2 * N_TRAIN_CLF, :]

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

    # Fit
    clf.fit(clf_tr.X, clf_tr.Y)

    tr_acc = eval(clf, clf_tr)
    print("Accuracy on training set: {:.2%}".format(tr_acc))

    # Test
    cl_ts = ts[:1000, :]
    ts_acc = eval(clf, cl_ts)
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
    x = sample.X[0, :]
    w = clf.forward(x)
    grad = clf.gradient(x, w=w)
    print(grad.shape)

    # Numerical gradient check
    from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases

    CClassifierTestCases.setUpClass()
    CClassifierTestCases()._test_gradient_numerical(clf, sample.X[0, :])

    print("done?")
