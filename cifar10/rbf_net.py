from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.rbf_net import CClassifierRBFNetwork, plot_train_curves, CClassifierRejectRBFNet

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets

import torch
from pyclustering.cluster.xmeans import xmeans

# PARAMETERS
SIGMA = 0.0
WD = 1e-10
EPOCHS = 250
BATCH_SIZE = 256
# FNAME = 'rbf_net_sigma_{:.3f}_{}'.format(SIGMA, EPOCHS)
FNAME = 'rbfnet_nr_like_wd_{:.0e}'.format(WD)


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

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    # HACK: SELECTING VALIDATION DATA (shape=2*N_TEST)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=2*N_TEST, random_state=random_state)
    vl_sample = ts[ts_idxs[:N_TEST], :]
    ts_sample = ts[ts_idxs[N_TEST:], :]

    # Create DNR
    # layers = ['features:23', 'features:26', 'features:29']
    # n_hiddens = [500, 300, 100]
    layers = ['features:29']

    # # X-Means Clustering for prototypes init.
    # feat_extr = CNormalizerDNN(dnn, out_layer=layers[-1])
    # feats = feat_extr.transform(tr_sample.X)
    # xm = xmeans(feats.tondarray())
    # xm.process()
    # n_hiddens = [len(xm.get_centers())]

    # n_hiddens = [100]

    # Init with NR support-vectors
    sv_nr = CArray.load('sv_nr')
    n_hiddens = [sv_nr.shape[0]]
    rbf_net = CClassifierRBFNetwork(dnn, layers,
                                    n_hiddens=n_hiddens,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=vl_sample,
                                    weight_decay=WD,
                                    sigma=SIGMA,              # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                    random_state=random_state)

    print("RBF network config:")
    for l, h in zip(layers, n_hiddens):
        print("{} -> {}".format(l, h))

    # =================== PROTOTYPE INIT. ===================

    # # Initialize prototypes with some training samples
    # h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    # idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    # proto = tr_sample.X[idxs, :]
    # rbf_net.prototypes = proto

    # # 1 prototype per class init.
    # proto = CArray.zeros((10, tr_sample.X.shape[1]))
    # for c in range(10):
    #     proto[c, :] = tr_sample.X[tr_sample.Y == c, :][0, :]
    # rbf_net.prototypes = proto

    # rbf_net._clf.model.prototypes = [torch.Tensor(xm.get_centers()).to('cuda')]

    feat_extr = CNormalizerDNN(dnn, out_layer=layers[-1])
    feats = feat_extr.transform(sv_nr.tondarray())
    rbf_net._clf.model.prototypes = [torch.Tensor(feats.tondarray()).to('cuda')]

    # =================== GAMMA INIT. ===================

    # # Rule of thumb 'gamma' init
    # gammas = []
    # for i in range(len(n_hiddens)):
    #     d = rbf_net._num_features[i].item()
    #     gammas.append(CArray([1/d] * n_hiddens[i]))
    # rbf_net.betas = gammas
    # Avoid training for betas
    # rbf_net.train_betas = False
    # print("-> Gamma init. with rule of thumb and NOT trained <-")

    print("Hyperparameters:")
    print("- sigma: {}".format(SIGMA))
    print("- weight_decay: {}".format(WD))
    print("- batch_size: {}".format(BATCH_SIZE))
    print("- epochs: {}".format(EPOCHS))

    print("\n Training:")
    # Fit DNR
    rbf_net.verbose = 2  # DEBUG
    rbf_net.fit(tr_sample.X, tr_sample.Y)
    rbf_net.verbose = 0

    # Plot training curves
    fig = plot_train_curves(rbf_net.history, SIGMA)
    fig.savefig("rbf_net_train_sigma_{:.3f}_curves.png".format(SIGMA))

    # Check test performance
    y_pred = rbf_net.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("RBFNet Accuracy: {}".format(acc))

    # We can now create a classifier with reject
    clf_rej = CClassifierRejectRBFNet(rbf_net, 0.)

    # Set threshold (FPR: 10%)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Dump to disk
    clf_rej.save(FNAME)
    print("Output file: {}.gz".format(FNAME))
