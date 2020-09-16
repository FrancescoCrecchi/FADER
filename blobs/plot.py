from secml.adv.seceval import CSecEval
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetricAccuracyReject

from plot_seval import compute_performance, acc_rej_performance, rej_percentage

fig = CFigure(height=10, width=16)
# Sec eval plot code
sp1 = fig.subplot(2, 1, 1)
sp2 = fig.subplot(2, 1, 2)


# CLFS = ['rbfnet', 'svm']
# CLFS = ['rbfnet_998', 'rbfnet_999']

# CLFS = ['rbfnet_degenere', 'rbfnet_attesa']
# FNAME = 'sec_eval_confronto_rbf'

# CLFS = ['svm', 'rbfnet_attesa']
# FNAME = 'sec_eval_confronto_rbf_svm'

# CLFS = ['rbfnet_attesa', 'rbfnet_cat_hinge', 'svm']
# FNAME = 'sec_eval_confronto_rbf_cat_hinge'

CLFS = ['svm', 'rbfnet_999']
FNAME = 'sec_eval_secml_pip'

print("Plotting")
for clf_name in CLFS:
    # Load data
    sec_eval = CSecEval.load('{}_sec_eval.gz'.format(clf_name))
    clf = sec_eval.attack.classifier
    sec_evals_data = sec_eval.sec_eval_data

    print(" - " + clf_name)

    # Plot performance
    _, perf, perf_std = compute_performance(sec_evals_data, acc_rej_performance)
    sp1.plot(sec_evals_data.param_values, perf, label=clf_name)
    # Plot mean and std
    std_up = perf + perf_std
    std_down = perf - perf_std
    std_down[std_down < 0.0] = 0.0
    std_down[std_up > 1.0] = 1.0
    sp1.fill_between(sec_evals_data.param_values, std_up, std_down, interpolate=False, alpha=0.2)

    # Plot reject percentage
    _, perf, perf_std = compute_performance(sec_evals_data, rej_percentage)
    sp2.plot(sec_evals_data.param_values, perf, label=clf_name)
    # Plot mean and std
    std_up = perf + perf_std
    std_down = perf - perf_std
    std_down[std_down < 0.0] = 0.0
    std_down[std_up > 1.0] = 1.0
    sp2.fill_between(sec_evals_data.param_values, std_up, std_down, interpolate=False, alpha=0.2)

sp1.xticks(sec_evals_data.param_values)
sp1.xlabel(sec_evals_data.param_name)
sp1.ylabel(CMetricAccuracyReject().class_type.capitalize())
sp1.legend()
sp1.title("Security Evaluation Curve")
sp1.apply_params_sec_eval()

# Rej percentage plot code
sp2.xticks(sec_evals_data.param_values)
sp2.xlabel(sec_evals_data.param_name)
sp2.ylabel("% Reject")
sp2.apply_params_sec_eval()

fig.savefig(FNAME)