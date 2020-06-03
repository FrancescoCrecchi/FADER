from secml.array import CArray
from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierDNR, CClassifierRejectThreshold

from mnist.fit_dnn import get_datasets
from mnist.wb_dnr_surrogate import CClassifierDNRSurrogate
from mnist.wb_nr_surrogate import CClassifierRejectSurrogate

random_state = 999
tr, _, ts = get_datasets(random_state)

# Load classifier and wrap it
# NR
# clf = CClassifierRejectThreshold.load('clf_rej.gz')
# clf = CClassifierRejectSurrogate(clf)

# DNR
clf = CClassifierDNR.load('dnr.gz')
clf = CClassifierDNRSurrogate(clf)

# Check test performance
y_pred = clf.predict(ts.X, return_decision_function=False)

from secml.ml.peval.metrics import CMetric
acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(acc_torch))

# Tune attack params
one_ds = ts[ts.Y == 1, :]
x0, y0 = one_ds[22, :].X, one_ds[22, :].Y

# Defining attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 5.0  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = 8  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.1,
    'eta_min': 0.1,
    'eta_pgd': 0.1,
    'max_iter': 100,
    'eps': 1e-10,
}
# solver_params = None
pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                  surrogate_classifier=clf,
                                  surrogate_data=one_ds,
                                  distance=noise_type,
                                  lb=lb, ub=ub,
                                  dmax=dmax,
                                  solver_params=solver_params,
                                  y_target=y_target)
pgd_attack.verbose = 2  # DEBUG

eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(x0, y0, double_init=True)
# assert eva_y_pred.item == 8, "Attack not working"

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

# Dump attack to disk
pgd_attack.verbose = 0
pgd_attack.y_target = None
pgd_attack.save('nr_wb_attack')
