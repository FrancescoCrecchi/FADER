import os

from secml.adv.seceval import CSecEval, CSecEvalData
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR

from mnist.rbf_net import CClassifierRejectRBFNet, CClassifierRBFNetwork


def transfer_attack(clf, seval_data):
    '''
    Performs transferability of seval to clf classifier
    :param clf: Classifier to be tested for transfer attack
    :param seval_data: Security evaluation data containing attack data
    :return: Security evaluation data for tranferability attack on clf
    '''

    # Main loop
    res = seval_data.deepcopy()
    for i, ds in enumerate(seval_data.adv_ds):
        pred, scores = clf.predict(ds.X, return_decision_function=True)
        res.scores[i] = scores
        res.Y_pred[i] = pred

    return res


DSET = 'mnist'
# CLFS = ['nr', 'dnr', 'tsne_rej', 'tnr']

if DSET == 'mnist':
    # MNIST Final
    CLFS = [
        'nr',
        'rbfnet_nr_like_10_wd_0e+00',
        'dnr',
        'dnr_rbf_tr_init'
        ]
elif DSET == 'cifar10':
    # CIFAR10 Final
    CLFS = [
        'nr',
        'rbf_net_nr_sv_100_wd_0e+00_cat_hinge_tr_init',
        'dnr',
        'dnr_rbf'
        ]
else:
    raise ValueError("Unrecognized dataset!")

N_ITER = 3
if __name__ == '__main__':
    random_state = 999

    # Load sevals
    dnn_sevals_data = []
    for i in range(N_ITER):
        seval = CSecEval.load(os.path.join(DSET, "dnn_wb_seval_it_{:d}.gz".format(i)))
        dnn_sevals_data.append(seval.sec_eval_data)

    # Run on classifiers
    print("Transfer to:")
    for _clf in CLFS:

        # Load clf
        if _clf == 'nr' or _clf == 'tsne_rej':
            clf = CClassifierRejectThreshold.load(os.path.join(DSET, _clf + '.gz'))
        elif 'dnr' in _clf or _clf == 'tnr':
            clf = CClassifierDNR.load(os.path.join(DSET, _clf + '.gz'))
        elif "rbf_net" in _clf or "rbfnet" in _clf:
            clf = CClassifierRejectRBFNet.load(os.path.join(DSET, _clf + '.gz'))
        else:
            raise ValueError("Unknown model to test for transferability!")

        print("- " + _clf)

        for i in range(N_ITER):
            print(" #{}".format(i), end='')
            # Transferability test
            transfer_seval = transfer_attack(clf, dnn_sevals_data[i])
            # Dump to disk
            transfer_seval.save(os.path.join(DSET, _clf + "_bb_seval_it_{:d}".format(i)))
        print()
