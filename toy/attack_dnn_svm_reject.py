from secml.adv.attacks import CAttackEvasionPGDExp
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml import CKernelRBF, CClassifierSVM
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.utils import fm

from toy.utils import get_cifar10_preprocess, get_datasets_cifar10, \
    plot_seceval_reject, get_accuracy

random_state = 999

# Load data
ds_tr, ds_vl, ds_ts = get_datasets_cifar10(random_state)

# CIFAR10 pre-trained Net Preprocessor
dnn_pre = get_cifar10_preprocess()

clf_norej = CClassifierSVM(
    C=1e-1, kernel=CKernelRBF(gamma=1e-2), preprocess=dnn_pre)

if True:
    clf_norej.fit(ds_vl.X, ds_vl.Y)

    get_accuracy(clf_norej, ds_ts)

    clf_norej.save_state(fm.join(fm.abspath(__file__), 'dnn_svm_state.gz'))

clf_norej.load_state(fm.join(fm.abspath(__file__), 'dnn_svm_state.gz'))

clf = CClassifierRejectThreshold(clf_norej, threshold=-0.95)

# clf.threshold = clf.compute_threshold(0.01, ds_vl)
print(clf.threshold)

# get_accuracy_reject(clf, ds_ts)

# Defining attack
noise_type = 'l2'
dmax = 2.1
lb, ub = 0., 1.
y_target = None

eps = CArray([0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0])

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.1,
    'eta_min': 0.1,
    'max_iter': 40,
    'eps': 1e-6
}

if True:

    pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                      double_init=False,
                                      distance=noise_type,
                                      lb=lb, ub=ub,
                                      dmax=dmax,
                                      solver_params=solver_params,
                                      y_target=y_target)
    pgd_attack.verbose = 1

    # Attack sample
    sample_idx = CArray.randsample(
        ds_ts.X.shape[0], shape=25, random_state=random_state)
    ds_adv = ds_ts[sample_idx, :]

    # Security evaluation
    sec_eval = CSecEval(attack=pgd_attack,
                        param_name='dmax', param_values=eps,
                        save_adv_ds=False)
    sec_eval.verbose = 2  # DEBUG

    clf_norej.verbose = 0

    # Run the security evaluation using the test set
    print("Running security evaluation...")
    sec_eval.run_sec_eval(ds_adv)
    print("Done!")

    # Save to disk
    sec_eval.save(fm.join(fm.abspath(__file__), 'dnn_svm_reject_seceval.gz'))

sec_eval = CSecEval.load(
    fm.join(fm.abspath(__file__), 'dnn_svm_reject_seceval.gz'))

plot_seceval_reject(sec_eval, eps, label='DNN-SVM (R)', name='dnn_svm')
