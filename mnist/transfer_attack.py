from secml.adv.attacks import CAttackEvasionPGD
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


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
    seval = CSecEval(attack=pgd_attack, param_name='dmax', param_values=evals, save_adv_ds=True)
    seval.verbose = 1  # DEBUG

    # Run the security evaluation using the test set
    print("Running security evaluation...")
    seval.run_sec_eval(dset, double_init=False)
    print("Done!")

    return seval


N_SAMPLES = 1000       # TODO: restore full dataset
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Load clf_rej
    clf_rej = CClassifierRejectThreshold.load('clf_rej.gz')
    # Set threshold
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts)

    # "Used to perturb all test samples"
    eps = CArray.arange(start=0, step=0.5, stop=5.1)
    sec_eval = security_evaluation(ts[:N_SAMPLES, :], clf_rej, dnn, tr, eps)

    # Save to disk
    sec_eval.save('clf_rej_bb_seval_v2')
