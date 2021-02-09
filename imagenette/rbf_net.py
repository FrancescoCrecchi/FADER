import sys
sys.path.append("/home/asotgiu/paper_neurocomputing/dnr")
sys.path.append("/home/asotgiu/paper_neurocomputing/secml-pip/src")
import matplotlib
matplotlib.use("Agg")

from imagenette.dataset_loading import load_imagenette, load_imagenet
from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CClassifierPyTorch
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.rbf_net import CClassifierRBFNetwork, plot_train_curves, \
    CClassifierRejectRBFNet

from torchvision.models import alexnet
import torch.nn as nn


# PARAMETERS
SIGMA = 0.0
WD = 0.0
EPOCHS = 250
BATCH_SIZE = 128

N_PROTO = 100
LOSS = 'cat_hinge' # 'xentr'

LAYER = 'classifier:5'

# FNAME = 'rbf_net_sigma_{:.3f}_{}'.format(SIGMA, EPOCHS)
# FNAME = 'rbfnet_nr_like_wd_{:.0e}'.format(WD)
# FNAME = 'rbf_net_nr_sv_{}_wd_{:.0e}_{}'.format(N_PROTO, WD, LOSS)
FNAME = 'rbf_net_{}_{}_wd_{:.0e}_{}'.format(LAYER, N_PROTO, WD, LOSS)


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

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    # HACK: SELECTING VALIDATION DATA (shape=2*N_TEST)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]
    vl_sample = ts.deepcopy()

    # Create DNR
    # layers = ['features:23', 'features:26', 'features:29']
    # n_hiddens = [500, 300, 100]
    layers = [LAYER]

    # # X-Means Clustering for prototypes init.
    # print("-> Prototypes: X-means initialization <-")
    # feat_extr = CNormalizerDNN(dnn, out_layer=layers[-1])
    # feats = feat_extr.transform(tr_sample.X)
    # xm = xmeans(feats.tondarray())
    # xm.process()
    # n_hiddens = [len(xm.get_centers())]

    # # Init with NR support-vectors
    # print("-> Prototypes: SVM support vectors initialization <-")
    # nr = CClassifierRejectThreshold.load('nr.gz')
    # sv_nr = tr_sample.X[nr.clf._sv_idx, :]      # Previously: sv_nr = CArray.load('sv_nr')
    # n_hiddens = [sv_nr.shape[0]]

    n_hiddens = [N_PROTO]
    rbf_net = CClassifierRBFNetwork(dnn, layers,
                                    n_hiddens=N_PROTO,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=vl_sample,
                                    loss=LOSS,
                                    weight_decay=WD,
                                    sigma=SIGMA,              # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                    random_state=random_state)

    print("RBF network config:")
    for l, h in zip(layers, n_hiddens):
        print("{} -> {}".format(l, h))

    # =================== PROTOTYPE INIT. ===================

    # Initialize prototypes with some training samples
    print("-> Prototypes: Training samples initialization <-")
    h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    proto = CArray.zeros((h, tr_sample.X.shape[1]))
    n_proto_per_class = h // dnn.n_classes
    for c in range(dnn.n_classes):
        proto[c*n_proto_per_class: (c+1)*n_proto_per_class, :] = tr_sample.X[tr_sample.Y == c, :][:n_proto_per_class, :]
    # idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    # proto = tr_sample.X[idxs, :]
    rbf_net.prototypes = proto

    # # 1 prototype per class init.
    # proto = CArray.zeros((10, tr_sample.X.shape[1]))
    # for c in range(10):
    #     proto[c, :] = tr_sample.X[tr_sample.Y == c, :][0, :]
    # rbf_net.prototypes = proto

    # rbf_net._clf.model.prototypes = [torch.Tensor(xm.get_centers()).to('cuda')]

    # # Init with NR support-vectors
    # print("-> Prototypes: SVM support vectors initialization <-")
    # nr = CClassifierRejectThreshold.load('nr.gz')
    # sv_nr = tr_sample[nr.clf._sv_idx, :]  # Previously: sv_nr = CArray.load('sv_nr')
    # # Reduce prototypes to the desired amount
    # proto = CArray.zeros((N_PROTO, sv_nr.X.shape[1]))
    # proto_per_class = N_PROTO // dnn.n_classes
    # for c in range(dnn.n_classes):
    #     proto[c * proto_per_class:(c + 1) * proto_per_class, :] = sv_nr.X[sv_nr.Y == c, :][:proto_per_class, :]
    # rbf_net.prototypes = proto

    # feat_extr = CNormalizerDNN(dnn, out_layer=layers[-1])
    # feats = feat_extr.transform(sv_nr.tondarray())
    # rbf_net._clf.model.prototypes = [torch.Tensor(feats.tondarray()).to('cuda')]

    # =================== GAMMA INIT. ===================

    # Rule of thumb 'gamma' init
    print("-> Gamma init. with rule of thumb <-")
    gammas = []
    for i in range(len(n_hiddens)):
        d = rbf_net._num_features[i].item()
        gammas.append(CArray([1/d] * n_hiddens[i]))
    rbf_net.betas = gammas
    # Avoid training for betas
    rbf_net.train_betas = False
    print("-> Gammas NOT trained <-")

    print("Hyperparameters:")
    print("- sigma: {}".format(SIGMA))
    print("- loss: {}".format(LOSS))
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
    # FNAME = os.path.join('ablation_study', FNAME)
    print("Output file: {}.gz".format(FNAME))
    clf_rej.save(FNAME)
