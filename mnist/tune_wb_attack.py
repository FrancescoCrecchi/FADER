from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject

from mnist.adv_reg_dnn import AdvNormRegClf, adv_mnist_cnn
from mnist.fit_dnn import get_datasets
from mnist.rbf_net import CClassifierRBFNetwork, CClassifierRejectRBFNet
from mnist.deep_rbf_net import CClassifierDeepRBFNetwork

from wb_dnr_surrogate import CClassifierDNRSurrogate
from wb_nr_surrogate import CClassifierRejectSurrogate

# TODO: Set this!
CLF = 'tsne_rej'
USE_SMOOTHING = False
N_SAMPLES = 10
N_PLOTS = 4

random_state = 999
_, vl, ts = get_datasets(random_state)

# Load classifier and wrap it
if CLF == 'nr' or CLF == 'tsne_rej':
    # NR
    clf = CClassifierRejectThreshold.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierRejectSurrogate(clf, gamma_smoothing=1000)
elif CLF == 'dnr' or CLF == 'tnr':
    # DNR
    clf = CClassifierDNR.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierDNRSurrogate(clf, gamma_smoothing=1000)
elif "deep_rbf_net" in CLF:
    # DEBUG: DUPLICATED CODE TO AVOID SMOOTHING
    clf = CClassifierRejectRBFNet.load(CLF + '.gz')
# elif "adv_reg_dnn" in CLF:
#     # Fit DNN
#     clf = adv_mnist_cnn()
#     clf.load_model(CLF + '.pkl')
else:
    raise ValueError("Unknown classifier!")
# clf.verbose = 2     # DEBUG

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
dmax = 3.0  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.1,
    'eta_min': 0.1,
    # 'eta_pgd': 0.01,
    'max_iter': 40,
    'eps': 1e-6
}
# solver_params = None
pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                  surrogate_classifier=clf,
                                  surrogate_data=tr_sample,
                                  distance=noise_type,
                                  lb=lb, ub=ub,
                                  dmax=dmax,
                                  solver_params=solver_params,
                                  y_target=y_target)
pgd_attack.verbose = 2  # DEBUG

# Attack N_SAMPLES
sample = ts[:N_SAMPLES, :]
# Plot N_PLOTS random attack samples
# sel_idxs = CArray.randsample(ts.X.shape[0], shape=N_SAMPLES, random_state=random_state)
# sample = ts[sel_idxs, :]

eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(sample.X, sample.Y)    # double_init=False

# Compute attack performance
assert dmax > 0, "Wrong dmax!"
perf = CMetricAccuracyReject().performance_score(y_true=sample.Y, y_pred=eva_y_pred)
print("Performance under attack: {0:.2f}".format(perf))

# # Plot N_PLOTS random attack samples
# sel_idxs = CArray.randsample(sample.X.shape[0], shape=N_PLOTS, random_state=random_state)
# selected = sample[sel_idxs, :]

# TODO: Select "not evading" samples!
not_evading_samples = sample[(eva_y_pred == sample.Y).logical_or(eva_y_pred == -1), :]
selected = not_evading_samples
# not_evading_samples.save("not_evading_wb_"+CLF)

N = min(selected.X.shape[0], N_PLOTS)
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

# Dump attack to disk
pgd_attack.verbose = 0
pgd_attack.save(CLF+'_wb_attack')
