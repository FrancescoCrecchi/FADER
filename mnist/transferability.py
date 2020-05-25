from secml.adv.seceval import CSecEval
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR

from mnist.fit_dnn import get_datasets

def transfer_attack(clf, seval):
    '''
    Performs transferability of seval to clf classifier
    :param clf: Classifier to be tested for transfer attack
    :param seval: Security evaluation containing attack data
    :return: seval for tranferability attack on clf
    '''

    res = seval.copy()

    # # Remove unnecessary params
    # res.sec_eval_data.fobj = None

    # Main loop
    for i, ds in enumerate(res.sec_eval_data.adv_ds):
        pred, scores = clf.predict(ds.X, return_decision_function=True)
        res.sec_eval_data.scores[i] = scores
        res.sec_eval_data.Y_pred[i] = pred

    return res


if __name__ == '__main__':
    random_state = 999

    # Load clf
    # clf_rej = CClassifierRejectThreshold.load('clf_rej.gz')
    dnr = CClassifierDNR.load('dnr.gz')

    # Load adversarial samples
    dnn_seval = CSecEval.load("dnn_seval.gz")

    # Transferability test
    # clf_rej_bb_seval = transfer_attack(clf_rej, dnn_seval)
    dnr_bb_seval = transfer_attack(dnr, dnn_seval)

    # Dump to disk
    # clf_rej_bb_seval.save("clf_rej_bb_seval")
    dnr_bb_seval.save("dnr_bb_seval")