from secml.adv.attacks import CAttackEvasionPGDExp
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CClassifierPyTorch
from secml.utils import fm

from toy.cnn_cifar10 import cifar10
from toy.utils import get_datasets_cifar10

random_state = 999

ds_tr, ds_vl, ds_ts = get_datasets_cifar10(random_state)

# Fit DNN
model = cifar10(pretrained=True)

dnn = CClassifierPyTorch(model,
                         pretrained=True,
                         input_shape=(3, 32, 32),
                         preprocess=CNormalizerMeanStd(
                             mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
                         n_jobs=2,
                         random_state=0)

# Defining attack
noise_type = 'l2'
dmax = 2.1
lb, ub = 0., 1.
y_target = None

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.1,
    'eta_min': 0.1,
    'max_iter': 40,
    'eps': 1e-6
}

pgd_attack = CAttackEvasionPGDExp(classifier=dnn,
                                  double_init=False,
                                  distance=noise_type,
                                  lb=lb, ub=ub,
                                  dmax=dmax,
                                  solver_params=solver_params,
                                  y_target=y_target)
pgd_attack.verbose = 1

# Attack sample
sample_idx = CArray.randsample(
    ds_ts.X.shape[0], shape=10, random_state=random_state)
ds_adv = ds_ts[sample_idx, :]

eps = CArray([0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0])

# Security evaluation
sec_eval = CSecEval(attack=pgd_attack,
                    param_name='dmax', param_values=eps,
                    save_adv_ds=True)
sec_eval.verbose = 2  # DEBUG

if False:

    # Run the security evaluation using the test set
    print("Running security evaluation...")
    sec_eval.run_sec_eval(ds_adv)
    print("Done!")

    # Save to disk
    sec_eval.save(fm.join(fm.abspath(__file__), 'dnn_seceval.gz'))

sec_eval = CSecEval.load(fm.join(fm.abspath(__file__), 'dnn_seceval.gz'))

from secml.figure import CFigure

fig = CFigure(height=5, width=5)

# Convenience function for plotting the Security Evaluation Curve
fig.sp.plot_sec_eval(
    sec_eval.sec_eval_data, marker='o', label='DNN', show_average=True)
fig.sp.xscale('symlog', linthreshx=0.1)
fig.sp.xticks(eps)
fig.sp.xticklabels(eps)
fig.sp.ylim(-0.05, 1.05)
fig.savefig(fm.join(fm.abspath(__file__), 'dnn_seceval_plot.pdf'))
