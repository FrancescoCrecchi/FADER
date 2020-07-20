import argparse

from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.attack_dnn import security_evaluation
from cifar10.fit_dnn import get_datasets


def attack_param_scheduler(seval):
    if seval.attack.dmax >= 1.0:
        # Should be chosen depending on the optimization problem
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'eta_pgd': 0.1,
            'max_iter': 40,
            'eps': 1e-8
        }
        seval._attack.set('solver_params', solver_params)


EPS = CArray.arange(start=0, step=0.05, stop=2.1)
if __name__ == '__main__':

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("clf", help="Model type", type=str)
    parser.add_argument("-n", "--n_samples", help="Number of attack samples to use for security evaluation", type=int, default=100)
    parser.add_argument("-i", "--iter", help="Number of independent iterations to performs", type=int, default=3)
    args = parser.parse_args()

    random_state = 999
    tr, _, ts = get_datasets(random_state)

    print("- Attacking ", args.clf)

    if args.clf == 'dnn':
        pgd_attack = CAttackEvasionPGD.load(args.clf + '_attack.gz')
    else:
        # Load attack
        pgd_attack = CAttackEvasionPGDExp.load(args.clf + '_wb_attack.gz')

    # DEBUG: remove!
    pgd_attack.n_jobs = 1

    # Check test performance
    clf = pgd_attack.surrogate_classifier
    y_pred = clf.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # HACK: Save attack parameters
    original_params = pgd_attack.solver_params.copy()

    for it in range(args.iter):

        print(" - It", str(it))
        # Select a sample of ts data
        it_idxs = CArray.randsample(ts.X.shape[0], shape=args.n_samples, random_state=random_state+it)
        ts_sample = ts[it_idxs, :]

        # "Used to perturb all test samples"
        sec_eval = security_evaluation(pgd_attack, ts_sample, EPS,
                                       pre_callbacks=[attack_param_scheduler])

        # Save to disk
        sec_eval.save(args.clf + '_wb_seval_it_' + str(it))

        # HACK: Restore original solver params
        pgd_attack.set('solver_params', original_params)
