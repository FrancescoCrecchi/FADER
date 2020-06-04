from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.attack_dnn import security_evaluation
from mnist.fit_dnn import get_datasets

CLFS = ['nr', 'dnr']
USE_DOUBLE_INIT = True

N_SAMPLES = 100     # TODO: restore full dataset
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    for _clf in CLFS:

        print("- Attacking ", _clf)

        # Load attack
        pgd_attack = CAttackEvasionPGDExp.load(_clf + '_wb_attack.gz')

        # Check test performance
        clf = pgd_attack.surrogate_classifier
        y_pred = clf.predict(ts.X, return_decision_function=False)
        acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
        print("Model Accuracy: {}".format(acc))

        # "Used to perturb all test samples"
        eps = CArray.arange(start=0, step=0.5, stop=5.1)
        sec_eval = security_evaluation(pgd_attack, ts[:N_SAMPLES, :], eps, double_init=USE_DOUBLE_INIT)

        # Save to disk
        sec_eval.save(_clf+'_wb_seval')