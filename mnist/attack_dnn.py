from secml.adv.attacks import CAttackEvasionPGD
from secml.adv.seceval import CSecEval
from secml.array import CArray

from mnist import mnist, get_datasets


def security_evaluation(dset, clf, surr, surr_dset, evals):
    # Defining attack
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    dmax = 3.0  # Maximum perturbation
    lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 1e-2,
        'max_iter': 50,
        'eps': 1e-4
    }
    pgd_attack = CAttackEvasionPGD(classifier=clf,
                                   surrogate_classifier=surr,
                                   surrogate_data=surr_dset,
                                   distance=noise_type,
                                   lb=lb, ub=ub,
                                   dmax=dmax,
                                   solver_params=solver_params,
                                   y_target=y_target)


    # Security evaluation
    seval = CSecEval(attack=pgd_attack, param_name='dmax', param_values=evals)
    seval.verbose = 1  # DEBUG

    # Run the security evaluation using the test set
    print("Running security evaluation...")
    seval.run_sec_eval(dset, double_init=False)
    print("Done!")

    return seval


if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    # Load classifier
    dnn = mnist()
    dnn.load_model('mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # "Used to perturb all test samples"
    eps = CArray.arange(start=0, step=0.5, stop=5.1)
    sec_eval = security_evaluation(ts, dnn, dnn, tr, eps)

    # Save to disk
    sec_eval.save_data('dnn_seval.pkl')
