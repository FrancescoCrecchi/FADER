import torch
from secml.array import CArray
from secml.data import CDataset
from secml.ml import CNormalizerMinMax
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.dnr_rbf import init_rbf_net, init_betas
from mnist.dnr_rbf import EPOCHS, BS, LOSS, WD
from mnist.fit_dnn import get_datasets

CLF = 'dnr_rbf_tr_init'

IN_DIM = 30
N_CLASSES = 10
N_HIDDENS = 100


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load DNR
    dnr = CClassifierDNR.load('{}.gz'.format(CLF))

    # Load scores dset
    scores_dset = CDataset.load('dnr_scores_dset.gz')

    # Replace combiner
    dnr._clf = init_rbf_net(IN_DIM, N_HIDDENS, N_CLASSES, random_state, EPOCHS, BS, LOSS, WD)
    # Normalize input scores
    # dnr._clf.preprocess = CNormalizerMinMax()

    # Restore threshold to default value
    dnr.threshold = -1000

    assert not dnr._clf.is_fitted(), "Something wrong here!"

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # =================== PROTOTYPE INIT. ===================
    #
    # print("-> Prototypes: Training samples initialization <-")
    # h = N_HIDDENS
    # proto = CArray.zeros((h, tr_sample.X.shape[1]))
    # n_proto_per_class = h // N_CLASSES
    # for c in range(N_CLASSES):
    #     proto[c * n_proto_per_class: (c + 1) * n_proto_per_class, :] = tr_sample.X[tr_sample.Y == c, :][
    #                                                                    :n_proto_per_class, :]
    #
    # # Compute scores dataset
    # # Hack: to make it work
    # dnr._clf._classes = CArray.arange(N_CLASSES)
    # f_x = dnr._get_layer_clfs_scores(proto)
    # # Set combiner prototypes
    # dnr._clf.model.prototypes = [torch.Tensor(f_x.tondarray()).to(dnr._clf._device)]

    # # =================== GAMMA INIT. ===================
    #
    # # Rule of thumb 'gamma' init
    # print("-> Gamma init. with rule of thumb <-")
    # init_betas(dnr.clf, N_HIDDENS, train_betas=False)
    # print("-> Gammas NOT trained <-")

    # Fit
    dnr._clf.verbose = 2 # DEBUG
    dnr._clf.fit(scores_dset.X, scores_dset.Y)
    dnr._clf.verbose = 0

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)

    # Dump to disk
    dnr.save(CLF)
