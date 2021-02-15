from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold, \
    CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets
from mnist.rbf_net import CClassifierRBFNetwork, CClassifierRejectRBFNet
from mnist.deep_rbf_net import CClassifierDeepRBFNetwork

from wb_dnr_surrogate import CClassifierDNRSurrogate
from wb_nr_surrogate import CClassifierRejectSurrogate


# Let's define a convenience function to easily plot the MNIST dataset
def show_digits(samples, preds, labels):
    samples = samples.atleast_2d()
    n_display = samples.shape[0]
    fig = CFigure(width=n_display*2, height=3)
    for idx in range(n_display):
        fig.subplot(2, n_display, idx+1)
        fig.sp.xticks([])
        fig.sp.yticks([])
        fig.sp.imshow(samples[idx, :].reshape((28, 28)), cmap='gray')
        fig.sp.title("{} ({})".format(labels[idx].item(), preds[idx].item()),
                     color=("green" if labels[idx].item()==preds[idx].item() else "red"))

    fig.savefig("tune_wb_attack_digits.png")

# TODO: Set this!
# CLF = 'dnn'
CLF = 'nr'
# CLF = 'rbfnet_nr_like_10_wd_0e+00'
# CLF = os.path.join('ablation_study', 'rbf_net_nr_sv_10_wd_0e+00')
# CLF = 'dnr'
# CLF = 'dnr_rbf_tr_init'

ATTACK = 'PGDExp'
# ATTACK = 'PGD'
# ATTACK = 'PGDMadri'

USE_SMOOTHING = False
N_SAMPLES = 1000
N_PLOTS = 0

random_state = 999
_, vl, ts = get_datasets(random_state)

# Load classifier and wrap it
if CLF == 'dnn':
    # Load classifier
    clf = cnn_mnist_model()
    clf.load_model('cnn_mnist.pkl')
    clf.verbose = 0
elif CLF == 'nr' or CLF == 'tsne_rej':
    # NR
    clf = CClassifierRejectThreshold.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierRejectSurrogate(clf, gamma_smoothing=1000)
    clf.verbose = 1     # INFO
elif 'dnr' in CLF or CLF == 'tnr':
    # DNR
    clf = CClassifierDNR.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierDNRSurrogate(clf, gamma_smoothing=1000)
    clf.verbose = 1     # INFO
elif "rbf_net" in CLF or "rbfnet" in CLF:
    # DEBUG: DUPLICATED CODE TO AVOID SMOOTHING
    if USE_SMOOTHING:
        print("WARNING: SMOOTHING ACTIVATED! (IGNORING)")
    clf = CClassifierRejectRBFNet.load(CLF + '.gz')
    clf.verbose = 1     # INFO
# elif "adv_reg_dnn" in CLF:
#     # Fit DNN
#     clf = adv_mnist_cnn()
#     clf.load_model(CLF + '.pkl')
else:
    raise ValueError("Unknown classifier!")

# Check test performance
y_pred = clf.predict(ts.X, return_decision_function=False)
perf = CMetricAccuracy().performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(perf))

# Select 10K training data and 1K test data (sampling)
N_TRAIN = 10000
tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
tr_sample = vl[tr_idxs, :]

# Defining attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 1.5  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # None if `error-generic` or a class label for `error-specific`

if ATTACK == 'PGDExp':

    if CLF == 'dnn':
        solver_params = {
            'eta': 1e-1,
            'eta_min': 0.1,
            'eta_pgd': 0.1,
            'max_iter': 100,
            'eps': 1e-4
        }
        double_init = True
    elif CLF == 'nr':
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'eta_pgd': 0.1,
            'max_iter': 100,
            'eps': 1e-4
        }
        double_init = True
    elif CLF in ('rbfnet_nr_like_10_wd_0e+00', ):
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            # 'eta_pgd': 0.1,
            'max_iter': 100,
            'eps': 1e-12
        }
        double_init = True
    elif CLF in ('dnr', ):
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'eta_pgd': 0.1,
            'max_iter': 100,
            'eps': 1e-12
        }
        double_init = True
    elif CLF in ('dnr_rbf_tr_init', ):
        solver_params = {
            'eta': 2,
            'eta_min': 2,
            # 'eta_pgd': 0.1,
            'max_iter': 100,
            'eps': 1e-12
        }
        double_init = True
    else:
        solver_params = None
        double_init = True

    pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                      double_init_ds=tr_sample,
                                      double_init=double_init,
                                      distance=noise_type,
                                      lb=lb, ub=ub,
                                      dmax=dmax,
                                      solver_params=solver_params,
                                      y_target=y_target)
    # pgd_attack = CAttackEvasionPGDExp.load(CLF+'_wb_attack.gz')
    # pgd_attack = pgd_attack.load(CLF+'_wb_attack.gz')
    pgd_attack.verbose = 1  # DEBUG
    # pgd_attack.dmax = 1.5
elif ATTACK == 'PGD':
    # Should be chosen depending on the optimization problem
    if CLF == 'dnn':
        solver_params = {
            'eta': 0.1,
            'max_iter': 100,
            'eps': 1e-4
        }
        double_init = False
    elif CLF in ('nr', ):
        solver_params = {
            'eta': 1e-1,
            'max_iter': 200,
            'eps': 1e-8
        }
        double_init = False
    elif CLF in ('rbfnet_nr_like_10_wd_0e+00', ):
        solver_params = {
            'eta': 1e-1,
            'max_iter': 200,
            'eps': 1e-8
        }
        double_init = False
    elif CLF in ('dnr', ):
        solver_params = {
            'eta': 1e-1,
            'max_iter': 200,
            'eps': 1e-12
        }
        double_init = False
    elif CLF in ('dnr_rbf_tr_init',):
        solver_params = {
            'eta': 1e-1,
            'max_iter': 200,
            'eps': 1e-4
        }
        double_init = False
    else:
        double_init = False
        solver_params = None
    # solver_params = None
    pgd_attack = CAttackEvasionPGD(classifier=clf,
                                   double_init_ds=tr_sample,
                                   double_init=double_init,
                                   distance=noise_type,
                                   lb=lb, ub=ub,
                                   dmax=dmax,
                                   solver_params=solver_params,
                                   y_target=y_target)
    pgd_attack.verbose = 2  # DEBUG
elif ATTACK == 'PGDMadri':
    # Should be chosen depending on the optimization problem
    if CLF == 'dnn':
        solver_params = {
            'eta': 2.5 * dmax / 100,
            'max_iter': 100,
            'eps': 1e-8
        }
        double_init = False
    elif CLF in ('nr', ):
        solver_params = {
            'eta': 2.5 * dmax / 100,
            'max_iter': 100,
            'eps': 1e-8
        }
        double_init = False
    elif CLF in ('rbfnet_nr_like_10_wd_0e+00', ):
        solver_params = {
            'eta': 2.5 * dmax / 100,
            'max_iter': 100,
            'eps': 1e-8
        }
        double_init = False
    elif CLF in ('dnr', ):
        solver_params = {
            'eta': 2.5 * dmax / 100,
            'max_iter': 100,
            'eps': 1e-12
        }
        double_init = False
    elif CLF in ('dnr_rbf_tr_init',):
        solver_params = {
            'eta': 2.5 * dmax / 100,
            'max_iter': 100,
            'eps': 1e-8
        }
        double_init = False
    else:
        double_init = False
        solver_params = None
    # solver_params = None
    pgd_attack = CAttackEvasionPGD(classifier=clf,
                                   double_init_ds=tr_sample,
                                   double_init=double_init,
                                   distance=noise_type,
                                   lb=lb, ub=ub,
                                   dmax=dmax,
                                   solver_params=solver_params,
                                   y_target=y_target)
    pgd_attack.verbose = 2  # DEBUG
else:
    raise ValueError("Unknown attack")

# Attack N_SAMPLES
# sample = ts[:N_SAMPLES, :]
# Plot N_SAMPLES random attack samples
sel_idxs = CArray.randsample(ts.X.shape[0], shape=N_SAMPLES, random_state=random_state)
sample = ts[sel_idxs, :]
eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(sample.X, sample.Y)    # double_init=False

print(eva_y_pred)
print(sample.Y)

# Compute attack performance
assert dmax > 0, "Wrong dmax!"
perf = CMetricAccuracyReject().performance_score(y_true=sample.Y, y_pred=eva_y_pred)
print("Performance under attack: {0:.2f}".format(perf))
# debug plot
# show_digits(eva_adv_ds.X, clf.predict(eva_adv_ds.X), eva_adv_ds.Y)

# # Plot N_PLOTS random attack samples
# sel_idxs = CArray.randsample(sample.X.shape[0], shape=N_PLOTS, random_state=random_state)
# selected = sample[sel_idxs, :]

# TODO: Select "not evading" samples!
not_evading_idxs = (eva_y_pred == sample.Y).logical_or(eva_y_pred == -1)
not_evading_samples = sample[not_evading_idxs, :]
selected = not_evading_samples
# not_evading_samples.save("not_evading_wb_"+CLF)

N = min(selected.X.shape[0], N_PLOTS)
if N > 0:
    fig = CFigure(height=5*N, width=16)
    for i in range(N):

        x0, y0 = selected[i, :].X, selected[i, :].Y

        # Rerun attack to have '_f_seq' and 'x_seq'
        _ = pgd_attack.run(x0, y0)

        # Loss curve
        sp1 = fig.subplot(N, 2, i*2+1)
        sp1.plot(pgd_attack._f_seq, marker='o', label='PGDExp')
        sp1.grid()
        sp1.xticks(range(pgd_attack._f_seq.shape[0]))
        sp1.xlabel('Iteration')
        sp1.ylabel('Loss')
        sp1.legend()

        # Confidence curves
        n_iter, n_classes = pgd_attack.x_seq.shape[0], clf.n_classes
        scores = CArray.zeros((n_iter, n_classes))
        for k in range(pgd_attack.x_seq.shape[0]):
            scores[k, :] = clf.decision_function(pgd_attack.x_seq[k, :])

        sp2 = fig.subplot(N, 2, i*2+2)
        for k in range(-1, clf.n_classes-1):
            sp2.plot(scores[:, k], marker='o', label=str(k))
        sp2.grid()
        sp2.xticks(range(pgd_attack.x_seq.shape[0]))
        sp2.xlabel('Iteration')
        sp2.ylabel('Confidence')
        sp2.legend()

    fig.savefig("wb_attack_tuning.png")
    # debug plot
    show_digits(eva_adv_ds[not_evading_idxs, :].X, clf.predict(eva_adv_ds[not_evading_idxs, :].X), eva_adv_ds[not_evading_idxs, :].Y)

# Dump attack to disk
pgd_attack.verbose = 0
pgd_attack.save(CLF+'_wb_'+ATTACK+'_attack')
