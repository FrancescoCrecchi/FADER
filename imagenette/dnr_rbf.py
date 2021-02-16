from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CNormalizerMinMax, CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

import torch
from torch import nn, optim
from torchvision.models import alexnet

from components.c_classifier_pytorch_rbf_network import \
    CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork, CategoricalHingeLoss
from imagenette.dataset_loading import load_imagenette, load_imagenet


EPOCHS = 250
BS = 64
LOSS = 'cat_hinge'  # 'xentr'
WD = 0.0
FNAME = 'dnr_rbf_tr_init'
N_JOBS = 1


def init_rbf_net(d, h, c, random_state, epochs, bs, loss, weight_decay):
    # Init DNN
    model = RBFNetwork(d, h, c)

    # Init betas
    model.betas = [torch.ones(h) * (1 / d)]
    # model.train_betas = False

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


def init_betas(rbfnet, h, train_betas=True):
    d = rbfnet.model.n_features[0]
    rbfnet.model.betas = [
        torch.Tensor(CArray([1 / d] * h).tondarray()).to(rbfnet._device)]
    rbfnet.model.train_betas = train_betas


N_TRAIN, N_TEST = 10000, 500
if __name__ == '__main__':
    random_state = 999

    vl = load_imagenette(exclude_val=True)
    ts = load_imagenet()

    # Load classifier
    net = alexnet(pretrained=True)
    linear = nn.Linear(in_features=4096, out_features=10, bias=True)
    linear.weight = nn.Parameter(
        net.classifier[-1].weight[
          [0, 217, 482, 491, 497, 566, 569, 571, 574, 701], :])
    linear.bias = nn.Parameter(
        net.classifier[-1].bias[
            [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]])
    net.classifier[-1] = linear
    dnn = CClassifierPyTorch(
        net, pretrained=True, input_shape=(3, 224, 224),
        preprocess=CNormalizerMeanStd(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create an RBF based DNR
    layers = ['classifier:3', 'classifier:4', 'classifier:5']

    layer_clf = {}
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    n_hiddens = [500, 200, 200]
    n_combiner = 100
    for i in range(len(layers)):
        layer_clf[layers[i]] = init_rbf_net(
            n_feats[i], n_hiddens[i], dnn.n_classes, random_state, EPOCHS,
            BS, LOSS, WD)

    combiner = init_rbf_net(dnn.n_classes*len(layers), n_combiner,
                            dnn.n_classes, random_state, EPOCHS, BS, LOSS, WD)
    # Normalizer Min-Max as preprocess
    combiner.preprocess = CNormalizerMinMax()

    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
    dnr.n_jobs = N_JOBS

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN,
                                random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST,
                                random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # =================== PROTOTYPE INIT. ===================
    # Initialize prototypes with some training samples
    print("-> Prototypes: Training samples initialization <-")
    h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    proto = CArray.zeros((h, tr_sample.X.shape[1]))
    n_proto_per_class = h // dnn.n_classes
    for c in range(dnn.n_classes):
        proto[c * n_proto_per_class: (c + 1) * n_proto_per_class, :] = \
            tr_sample.X[tr_sample.Y == c, :][:n_proto_per_class, :]
    # Compute
    # comb_proto = CArray.zeros((sum(n_hiddens), len(dnr._layers) * dnn.n_classes))
    # count = 0
    for i in range(len(layers)):
        lcf = layer_clf[layers[i]]
        f_x = lcf.preprocess.transform(proto[:n_hiddens[i], :])
        lcf.model.prototypes = [torch.Tensor(f_x.tondarray()).to(lcf._device)]
    #     # Store for combiner
    #     comb_proto[count: count+n_hiddens[i], :] = lcf.predict(proto[:n_hiddens[i], :])
    #     count += n_hiddens[i]
    # combiner.model.prototypes = [torch.Tensor(comb_proto.tondarray()).to(combiner._device)]

    # =================== GAMMA INIT. ===================

    # Rule of thumb 'gamma' init
    print("-> Gamma init. with rule of thumb <-")
    for i in range(len(n_hiddens)):
        lcf = dnr._layer_clfs[dnr._layers[i]]
        init_betas(lcf, n_hiddens[i], train_betas=False)
    init_betas(dnr.clf, n_combiner, train_betas=False)
    print("-> Gammas NOT trained <-")

    print("Hyperparameters:")
    # print("- sigma: {}".format(SIGMA))
    print("- loss: {}".format(LOSS))
    print("- weight_decay: {}".format(WD))
    print("- batch_size: {}".format(BS))
    print("- epochs: {}".format(EPOCHS))

    print("\n Training:")

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
    print("Output file: {}.gz".format(FNAME))
    dnr.save(FNAME)
