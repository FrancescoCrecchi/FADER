from secml.adv.attacks import CAttackEvasionPGD
from secml.figure import CFigure
from secml.ml import CNormalizerMeanStd
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets

random_state = 999
tr, _, ts = get_datasets(random_state)

# Load classifier
dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
dnn.load_model('cnn_cifar10.pkl')

# Check test performance
y_pred = dnn.predict(ts.X, return_decision_function=False)
acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(acc))

# Tune attack params
one_ds = ts.Y == 1
dbg = ts[one_ds, :]
x0, y0 = dbg[22, :].X, dbg[22, :].Y

# Defining attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 2.0  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = 7  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 5e-3,
    'max_iter': 50,
    'eps': 1e-4
}
# solver_params = None
pgd_attack = CAttackEvasionPGD(classifier=dnn,
                               surrogate_classifier=dnn,
                               surrogate_data=tr,
                               distance=noise_type,
                               lb=lb, ub=ub,
                               dmax=dmax,
                               solver_params=solver_params,
                               y_target=y_target)

pgd_attack.verbose = 2
eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(x0, y0, double_init=False)
# assert eva_y_pred.item == 8, "Attack not working"

# Plot attack loss function
fig = CFigure(height=5, width=10)
fig.sp.plot(pgd_attack._f_seq, marker='o', label='PGD')
fig.sp.grid()
fig.sp.xticks(range(pgd_attack._f_seq.shape[0]))
fig.sp.xlabel('Iteration')
fig.sp.ylabel('Loss')
fig.sp.legend()
fig.savefig("attack_loss.png")

# Plot confidence during attack
one_score, seven_score = [], []

for i in range(pgd_attack.x_seq.shape[0]):
    s = dnn.decision_function(pgd_attack.x_seq[i,:])
    one_score.append(s[1].item())
    seven_score.append(s[7].item())

fig = CFigure(height=5, width=10)
fig.sp.plot(one_score, marker='o', label='airplane')
fig.sp.plot(seven_score, marker='o', label='horse')
fig.sp.grid()
fig.sp.xticks(range(pgd_attack.x_seq.shape[0]))
fig.sp.xlabel('Iteration')
fig.sp.ylabel('Confidence')
fig.sp.legend()
fig.savefig("attack_confidence.png")

# Dump attack to disk
pgd_attack.verbose = 0
pgd_attack.y_target = None
pgd_attack.save('dnn_attack')
