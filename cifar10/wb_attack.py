from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.attack_dnn import security_evaluation
from cifar10.fit_dnn import get_datasets

CLFS = ['dnn']

N_SAMPLES = 100     # TODO: restore full dataset
ITER = 3
EPS = CArray.arange(start=0, step=0.05, stop=2.1)
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    for _clf in CLFS:

        print("- Attacking ", _clf)

        if _clf == 'dnn':
            pgd_attack = CAttackEvasionPGD.load(_clf + '_attack.gz')
        else:
            # Load attack
            pgd_attack = CAttackEvasionPGDExp.load(_clf + '_wb_attack.gz')

        # Check test performance
        clf = pgd_attack.surrogate_classifier
        y_pred = clf.predict(ts.X, return_decision_function=False)
        acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
        print("Model Accuracy: {}".format(acc))

        for it in range(ITER):
            print(" - It", str(it))
            # Select a sample of ts data
            it_idxs = CArray.randsample(ts.X.shape[0], shape=N_SAMPLES, random_state=random_state+it)
            ts_sample = ts[it_idxs, :]

            # "Used to perturb all test samples"
            sec_eval = security_evaluation(pgd_attack, ts_sample, EPS)

            # Save to disk
            sec_eval.save(_clf + '_wb_seval_it_' + str(it))