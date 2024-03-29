from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.array import CArray
from secml.figure import CFigure
from secml.ml import CNormalizerMeanStd, CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject

from torchvision.models import alexnet
import torch.nn as nn

from dataset_loading import load_imagenette, load_imagenet
from mnist.rbf_net import CClassifierRejectRBFNet

from wb_dnr_surrogate import CClassifierDNRSurrogate
from wb_nr_surrogate import CClassifierRejectSurrogate

# TODO: Set this!
# CLF = 'dnn'
# CLF = 'nr'
# CLF = 'dnr'
CLF = 'dnr_rbf_tr_init'
# CLF = 'rbf_net_classifier:5_100_wd_0e+00_cat_hinge'

ATTACK = 'PGDExp'
# ATTACK = 'PGD'
# ATTACK = 'PGDMadri'

USE_SMOOTHING = False
N_SAMPLES = 100
N_PLOTS = 10

random_state = 999
vl = load_imagenette(exclude_val=True)
ts = load_imagenet()

# Load classifier and wrap it
if CLF == 'dnn':
    net = alexnet(pretrained=True)
    linear = nn.Linear(in_features=4096, out_features=10, bias=True)
    linear.weight = nn.Parameter(
        net.classifier[-1].weight[
        [0, 217, 482, 491, 497, 566, 569, 571, 574, 701], :])
    linear.bias = nn.Parameter(
        net.classifier[-1].bias[
            [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]])
    net.classifier[-1] = linear
    clf = CClassifierPyTorch(
        net, pretrained=True, input_shape=(3, 224, 224),
        preprocess=CNormalizerMeanStd(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))
elif CLF == 'nr' or CLF == 'tsne_rej':
    # NR
    clf = CClassifierRejectThreshold.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierRejectSurrogate(clf, gamma_smoothing=10)
elif 'dnr' in CLF or CLF == 'tnr':
    # DNR
    clf = CClassifierDNR.load(CLF+'.gz')
    if USE_SMOOTHING:
        clf = CClassifierDNRSurrogate(clf, gamma_smoothing=10)
elif "rbf_net" in CLF or "rbfnet" in CLF:
    # DEBUG: DUPLICATED CODE TO AVOID SMOOTHING
    if USE_SMOOTHING:
        print("WARNING: SMOOTHING ACTIVATED! (IGNORING)")
    clf = CClassifierRejectRBFNet.load(CLF + '.gz')
elif "hybrid" in CLF:
    # DEBUG: DUPLICATED CODE TO AVOID SMOOTHING
    if USE_SMOOTHING:
        print("WARNING: SMOOTHING ACTIVATED! (IGNORING)")
    clf = CClassifierDNR.load(CLF + '.gz')
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
noise_type = 'l2'   # Type of perturbation 'l1' or 'l2'
dmax = 1.0          # Maximum perturbation
lb, ub = 0., 1.     # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None     # None if `error-generic` or a class label for `error-specific`

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

# Dump attack to disk
pgd_attack.verbose = 0
pgd_attack.save(CLF+'_wb_'+ATTACK+'_attack')
