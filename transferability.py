import os

from secml.adv.seceval import CSecEval, CSecEvalData
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR


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
CLFS = ['nr', 'dnr', 'tsne_rej', 'tnr']
N_ITER = 3
if __name__ == '__main__':
    random_state = 999

    # Load sevals
    dnn_sevals_data = []
    for i in range(N_ITER):
        seval = CSecEval.load(os.path.join(DSET, "dnn_wb_seval_it_{:d}.gz".format(i)))
        dnn_sevals_data.append(seval.sec_eval_data)

    # Run on classifiers
    for _clf in CLFS:

        # Load clf
        if _clf == 'nr' or _clf == 'tsne_rej':
            clf = CClassifierRejectThreshold.load(os.path.join(DSET, _clf + '.gz'))
        elif _clf == 'dnr' or _clf == 'tnr':
            clf = CClassifierDNR.load(os.path.join(DSET, _clf + '.gz'))
        else:
            raise ValueError("Unknown model to test for transferability!")
        clf.n_jobs = 16

        print("- Transfer to ", _clf)
        for i in range(N_ITER):
            print(" - It. ", str(i))
            # Transferability test
            transfer_seval = transfer_attack(clf, dnn_sevals_data[i])
            # Dump to disk
            transfer_seval.save(os.path.join(DSET, _clf + "_bb_seval_it_{:d}".format(i)))
