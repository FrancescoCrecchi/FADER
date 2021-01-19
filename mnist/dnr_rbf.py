import torch
from secml.array import CArray
from secml.ml import CNormalizerMinMax
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.dnr_rbf import init_rbf_net, init_betas
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

EPOCHS = 250
BS = 256
LOSS = 'cat_hinge' # 'xentr'
WD = 0.0
FNAME = 'dnr_rbf_tr_init'
N_JOBS = 1


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create an RBF based DNR
    layers = ['features:relu2', 'features:relu3', 'features:relu4']

    layer_clf = {}
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    n_hiddens = [250, 250, 50] #10x
    # n_hiddens = [1000, 1000, 300]  #2x
    n_combiner = 100
    for i in range(len(layers)):
        layer_clf[layers[i]] = init_rbf_net(n_feats[i], n_hiddens[i], dnn.n_classes, random_state, EPOCHS, BS, LOSS, WD)

    combiner = init_rbf_net(dnn.n_classes*len(layers), n_combiner, dnn.n_classes, random_state, EPOCHS, BS, LOSS, WD)

    # Normalizer Min-Max as preprocess
    combiner.preprocess = CNormalizerMinMax()

    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
    dnr.n_jobs = N_JOBS

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # =================== PROTOTYPE INIT. ===================
    # Initialize prototypes with some training samples
    print("-> Prototypes: Training samples initialization <-")
    h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    proto = CArray.zeros((h, tr_sample.X.shape[1]))
    n_proto_per_class = h // dnn.n_classes
    for c in range(dnn.n_classes):
        proto[c * n_proto_per_class: (c + 1) * n_proto_per_class, :] = tr_sample.X[tr_sample.Y == c, :][:n_proto_per_class, :]
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
