from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CKernelRBF
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.classifiers.sklearn.c_classifier_svm_m import CClassifierSVMM
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets
from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork

import torch
from torch import nn, optim

EPOCHS = 250
BS = 32

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

    # Create Hybrid DNR
    layers = ['features:23', 'features:26', 'features:29']

    # # 1) SVM-RBF + RBFNET
    # layer_clf = CClassifierSVMM(kernel=CKernelRBF(gamma=1), C=1)
    # layer_clf.n_jobs = 10       # DEBUG: SPEED-UP THINGS
    # rbfnet = RBFNetwork(30, 100, 10)
    # rbfnet.betas = [torch.ones(100) * 1/30]
    # combiner = CClassifierPyTorchRBFNetwork(rbfnet,
    #                                         loss=nn.CrossEntropyLoss(),
    #                                         optimizer=optim.Adam(rbfnet.parameters()),
    #                                         input_shape=(30,),
    #                                         epochs=EPOCHS,
    #                                         batch_size=BS,
    #                                         random_state=random_state)
    # dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
    #
    # '''
    #     Setting classifiers parameters (A. Sotgiu)
    #     CIFAR		    C	    gamma
    #     -----------------------------
    #     combiner	    1e-4	1
    #     features:23	    10	    1e-3
    #     features:26	    1	    1e-3
    #     features:29	    1e-1	1e-2
    #     '''
    # dnr.set_params({
    #     'features:23.C': 10,
    #     'features:23.kernel.gamma': 1e-3,
    #     'features:26.C': 1,
    #     'features:26.kernel.gamma': 1e-3,
    #     'features:29.C': 1e-1,
    #     'features:29.kernel.gamma': 1e-2,
    #     # 'clf.C': 1e-4,
    #     # 'clf.kernel.gamma': 1
    # })

    # 2) RBFNET + SVM-RBF
    layer_clf = []
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    n_hiddens = [500, 300, 100]
    for i in range(len(layers)):
        model = RBFNetwork(n_feats[i], n_hiddens[i], 10)

        # Init betas
        model.betas = [torch.ones(n_hiddens[i]) * (1/n_feats[i])]
        model.train_betas = False

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
        layer_clf.append(CClassifierPyTorchRBFNetwork(model,
                                                      loss=loss,
                                                      optimizer=optimizer,
                                                      input_shape=(n_feats[i],),
                                                      epochs=EPOCHS,
                                                      batch_size=BS,
                                                      random_state=random_state))

    combiner = CClassifierSVMM(kernel=CKernelRBF(gamma=1e-2), C=1)  # External xval (FC)
    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 2 # DEBUG
    for lcf in dnr._layer_clfs.values():
        lcf.verbose = 2  # DEBUG

    dnr.fit(tr_sample.X, tr_sample.Y)

    for lcf in dnr._layer_clfs.values():
        lcf.verbose = 0  # END DEBUG
    dnr.verbose = 0  # END DEBUG

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('hybrid_rbfnet_svm')
