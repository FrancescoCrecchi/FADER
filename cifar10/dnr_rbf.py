from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CNormalizerMinMax
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

import torch
from torch import nn, optim

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets
from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork, CategoricalHingeLoss

EPOCHS = 250
BS = 256
LOSS = 'xentr' # 'cat_hinge'
WD = 0.0


def init_rbf_net(d, h, c, random_state, epochs, bs, loss, weight_decay):
    # Init DNN
    model = RBFNetwork(d, h, c)

    # Init betas
    model.betas = [torch.ones(h) * (1 / d)]
    model.train_betas = False

    # Loss & Optimizer
    if loss == 'xentr':
        loss = nn.CrossEntropyLoss()
    elif loss == 'cat_hinge':
        loss = CategoricalHingeLoss(c)
    else:
        raise ValueError("Not a valid loss!")
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)

    return CClassifierPyTorchRBFNetwork(model,
                                        loss=loss,
                                        optimizer=optimizer,
                                        input_shape=(d,),
                                        epochs=epochs,
                                        batch_size=bs,
                                        random_state=random_state)


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

    # Create an RBF based DNR
    layers = ['features:23', 'features:26', 'features:29']

    layer_clf = {}
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    n_hiddens = [500, 300, 100]
    for i in range(len(layers)):
        layer_clf[layers[i]] = init_rbf_net(n_feats[i], n_hiddens[i], dnn.n_classes, random_state, EPOCHS, BS, LOSS, WD)

    combiner = init_rbf_net(dnn.n_classes*len(layers), 100, dnn.n_classes, random_state, EPOCHS, BS, LOSS, WD)
    # Normalizer Min-Max as preprocess
    combiner.preprocess = CNormalizerMinMax()

    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 2  # DEBUG
    for lcf in dnr._layer_clfs.values():
        lcf.verbose = 2  # DEBUG
    dnr.clf.verbose = 2  # DEBUG

    dnr.fit(tr_sample.X, tr_sample.Y)

    for lcf in dnr._layer_clfs.values():
        lcf.verbose = 0  # END DEBUG
    dnr.verbose = 0  # END DEBUG
    dnr.clf.verbose = 0  # END DEBUG

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('dnr_rbf')
