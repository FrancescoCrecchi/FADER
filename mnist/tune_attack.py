from secml.adv.attacks import CAttackEvasionPGD
from secml.figure import CFigure

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


random_state = 999
tr, _, ts = get_datasets(random_state)

# Load classifier
dnn = cnn_mnist_model()
dnn.load_model('cnn_mnist.pkl')

# Check test performance
y_pred = dnn.predict(ts.X, return_decision_function=False)

from secml.ml.peval.metrics import CMetric
acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(acc_torch))

# Tune attack params
one_eight = (ts.Y == 1).logical_or(ts.Y == 8)
dbg = ts[one_eight, :]
x0, y0 = dbg[0, :].X, dbg[0, :].Y

# Defining attack
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 3.0  # Maximum perturbation
lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = 8  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 1e-2,
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
one_score, eight_score = [], []

for i in range(pgd_attack.x_seq.shape[0]):
    s = dnn.decision_function(pgd_attack.x_seq[i,:])
    one_score.append(s[1].item())
    eight_score.append(s[8].item())

fig = CFigure(height=5, width=10)
fig.sp.plot(one_score, marker='o', label='1')
fig.sp.plot(eight_score, marker='o', label='8')
fig.sp.grid()
fig.sp.xticks(range(pgd_attack.x_seq.shape[0]))
fig.sp.xlabel('Iteration')
fig.sp.ylabel('Confidence')
fig.sp.legend()
fig.savefig("attack_confidence.png")

# TODO: Dump attack to disk


