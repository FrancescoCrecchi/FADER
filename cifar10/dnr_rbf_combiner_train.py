from secml.array import CArray
from secml.data import CDataset
from secml.ml import CNormalizerMinMax
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.dnr_rbf import init_rbf_net
from cifar10.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load DNR
    dnr = CClassifierDNR.load('dnr_rbf.gz.old')

    # Load scores dset
    scores_dset = CDataset.load('dnr_scores_dset.gz')

    # Normalize
    nmz = CNormalizerMinMax()
    scores_dset.X = nmz.fit_transform(scores_dset.X)

    # Replace combiner
    dnr._clf = init_rbf_net(30, 100, 10, random_state, 250, 32)
    assert not dnr._clf.is_fitted(), "Something wrong here!"

    # Fit
    dnr._clf.verbose = 2 # DEBUG
    dnr._clf.fit(scores_dset.X, scores_dset.Y)
    dnr._clf.verbose = 0

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    ## Select 10K training data and 1K test data (sampling)
    # tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    # tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)

    # Dump to disk
    dnr.save('dnr_rbf')



