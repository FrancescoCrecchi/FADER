from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.utils.c_file_manager import join

from components.monitor import CClassifierMonitored
from mnist.rbf_net import CClassifierRejectRBFNet, CClassifierRBFNetwork

DATASET = 'mnist'

# CLF = 'nr'
CLF = 'rbfnet_nr_like_10_wd_0e+00' if DATASET == 'mnist' else 'rbf_net_nr_sv_100_wd_0e+00_cat_hinge_tr_init'
# CLF = 'dnr'
# CLF = 'dnr_rbf_tr_init'

N_SAMPLES = 5000
N_REP = 5

if __name__ == '__main__':
    # Load data
    random_state = 999

    if DATASET == 'mnist':
        from mnist.fit_dnn import get_datasets
    elif DATASET == 'cifar10':
        from cifar10.fit_dnn import get_datasets
    else:
        ValueError("Unrecognized dataset!")

    tr, _, _ = get_datasets(random_state)

    # Load (pre-trained) classifier
    CPATH = join(DATASET, CLF + '.gz')
    if CLF == 'nr':
        # NR
        clf = CClassifierRejectThreshold.load(CPATH)
    elif 'dnr' in CLF:
        # DNR
        clf = CClassifierDNR.load(CPATH)
    elif "rbf_net" in CLF or "rbfnet" in CLF or 'fader' in CLF:
        clf = CClassifierRejectRBFNet.load(CPATH)
        # DEBUG: Moving _clf to cpu
        clf._clf._clf.to('cpu')
        # DEBUG: Checking batch_size
        # clf._clf._clf._batch_size = 32
        # clf._clf._clf._batch_size = 256
        # clf._clf._clf._batch_size = 1024
        clf._clf._clf._batch_size = 2048
        # clf._clf._clf._batch_size = 5096
    else:
        raise ValueError("Unknown classifier!")
    clf._check_is_fitted()

    # Measure times
    monitored = CClassifierMonitored(clf)
    monitored.verbose = 1

    for k in range(N_REP):
        _X = tr.X[k*N_SAMPLES:(k+1)*N_SAMPLES, :]
        monitored.predict(_X)

    timing = monitored.logtime_data
    print("{0}: Average time: {1:.2f}({2:.2f}) ms".format(CLF, timing.mean(), timing.std()))
