from secml.adv.seceval import CSecEval
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR


def transfer_attack(clf, seval):
    '''
    Performs transferability of seval to clf classifier
    :param clf: Classifier to be tested for transfer attack
    :param seval: Security evaluation containing attack data
    :return: seval for tranferability attack on clf
    '''

    res = seval.copy()

    # TODO: Remove unnecessary params
    # res.sec_eval_data.fobj = None

    # Main loop
    for i, ds in enumerate(res.sec_eval_data.adv_ds):
        pred, scores = clf.predict(ds.X, return_decision_function=True)
        res.sec_eval_data.scores[i] = scores
        # TODO: CHECK SCORE FOR NATURAL CLASSES
        res.sec_eval_data.Y_pred[i] = pred

    return res


CLF = 'tnr'
if __name__ == '__main__':
    random_state = 999

    # Load clf
    clf = CClassifierDNR.load(CLF+'.gz')

    # Load adversarial samples
    dnn_seval = CSecEval.load("dnn_seval.gz")

    # Transferability test
    transfer_seval = transfer_attack(clf, dnn_seval)

    # Dump to disk
    transfer_seval.save(CLF+"_bb_seval")