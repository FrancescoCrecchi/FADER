from secml.array import CArray
from secml.adv.attacks import CAttackEvasionPGDExp
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierDNR, CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy

from mnist.fit_dnn import get_datasets
from wb_dnr_surrogate import CClassifierDNRSurrogate
from wb_nr_surrogate import CClassifierRejectSurrogate

# TODO: Set this!
CLF = 'dnr'

random_state = 999
_, vl, ts = get_datasets(random_state)

# Load classifier and wrap it
if CLF == 'nr':
    # NR
    clf = CClassifierRejectThreshold.load('nr.gz')
    clf = CClassifierRejectSurrogate(clf, gamma_smoothing=1000)
elif CLF == 'dnr':
    # DNR
    clf = CClassifierDNR.load('dnr.gz')
    clf = CClassifierDNRSurrogate(clf, gamma_smoothing=1000)
elif CLF == 'tsne_rej':
    # TNR
    clf = CClassifierRejectThreshold.load('tsne_rej.gz')
    clf = CClassifierRejectSurrogate(clf, gamma_smoothing=1000)
else:
    raise ValueError("Unknown classifier!")

# Check test performance
y_pred = clf.predict(ts.X, return_decision_function=False)
acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(acc))

N_TRAIN = 10000
# Select 10K training data and 1K test data (sampling)
tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
tr_sample = vl[tr_idxs, :]

# Tune attack params
x0, y0 = ts[0, :].X, ts[0, :].Y

# Defining attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 5.0  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.1,
    'eta_min': 0.1,
    # 'eta_pgd': 0.1,
    'max_iter': 100,
    'eps': 1e-10,
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

eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(x0, y0)#, double_init=False)

# Plot attack loss function
fig = CFigure(height=5, width=10)
fig.sp.plot(pgd_attack._f_seq, marker='o', label='PGDExp')
fig.sp.grid()
fig.sp.xticks(range(pgd_attack._f_seq.shape[0]))
fig.sp.xlabel('Iteration')
fig.sp.ylabel('Loss')
fig.sp.legend()
fig.savefig("wb_attack_loss.png")

# Plot confidence during attack
n_iter, n_classes = pgd_attack.x_seq.shape[0], clf.n_classes
scores = CArray.zeros((n_iter, n_classes))

for i in range(pgd_attack.x_seq.shape[0]):
    scores[i, :] = clf.decision_function(pgd_attack.x_seq[i, :])

fig = CFigure(height=5, width=10)
for i in range(-1, clf.n_classes-1):
    fig.sp.plot(scores[:, i], marker='o', label=str(i))
fig.sp.grid()
fig.sp.xticks(range(pgd_attack.x_seq.shape[0]))
fig.sp.xlabel('Iteration')
fig.sp.ylabel('Confidence')
fig.sp.legend()
fig.savefig("wb_attack_confidence.png")

# # Dump attack to disk
# pgd_attack.verbose = 0
# pgd_attack.save(CLF+'_wb_attack')
