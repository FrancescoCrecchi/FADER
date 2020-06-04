from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.attack_dnn import security_evaluation
from mnist.fit_dnn import get_datasets

CLFS = ['nr', 'dnr']

N_SAMPLES = 100     # TODO: restore full dataset
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    for _clf in CLFS:
        # Load classifier and attack
        if _clf == 'nr':
            clf = CClassifierRejectThreshold.load('nr.gz')
        elif _clf == 'dnr':
            clf = CClassifierDNR.load('dnr.gz')
        else:
            ValueError('Unknown classifier to load!')

        print("- Attacking ", _clf)

        # Check test performance
        y_pred = clf.predict(ts.X, return_decision_function=False)
        acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
        print("Model Accuracy: {}".format(acc))

        # Load attack
        pgd_attack = CAttackEvasionPGDExp.load(_clf+'_wb_attack.gz')

        # "Used to perturb all test samples"
        eps = CArray.arange(start=0, step=0.5, stop=5.1)
        sec_eval = security_evaluation(pgd_attack, ts[:N_SAMPLES, :], eps)

        # Save to disk
        sec_eval.save(_clf+'_wb_seval')