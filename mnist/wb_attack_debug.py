from secml.adv.attacks import CAttackEvasionPGDExp
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetricAccuracyReject

from mnist.attack_dnn import security_evaluation
from mnist.plot_seval import compute_performance, rej_percentage

CLF = 'dnr'
# seval = CSecEval.load(CLF + '_wb_seval.gz')
#
# # Investigating non evading samples for largest attack magnitude
# orig_labels = seval.sec_eval_data.Y
# pred_labels = seval.sec_eval_data.Y_pred[-1]
#
# orig_samples = seval.sec_eval_data.adv_ds[0]    # EPS = 0
# not_evading_samples = orig_samples[(pred_labels == orig_labels).logical_or(pred_labels == -1), :]
# not_evading_samples.save("not_evading_wb_"+CLF)

# Load attack
pgd_attack = CAttackEvasionPGDExp.load(CLF + '_wb_attack.gz')
# # HACK: Reducing 'max_iter'
# pgd_attack._solver.max_iter = 30

# Load not evading samples from previous attack version
not_evading_samples = CDataset.load("not_evading_wb_"+CLF+".gz")

# Perform sec_eval on non-evading samples
eps = CArray.arange(start=3.5, step=0.5, stop=5.1)
sec_eval = security_evaluation(pgd_attack, not_evading_samples, eps)

# # Save to disk
# sec_eval.save(CLF+'_wb_new_attack')

# Plot performance
fig = CFigure(height=8, width=10)
# Sec eval plot code
sp1 = fig.subplot(2, 1, 1)
sp2 = fig.subplot(2, 1, 2)

# Plot performance
sp1.plot(sec_eval.sec_eval_data.param_values, compute_performance(sec_eval.sec_eval_data), label=CLF,
         linestyle='-', color=None, marker='o',
         markevery=sp1.get_xticks_idx(sec_eval.sec_eval_data.param_values))
# Plot reject percentage
sp2.plot(sec_eval.sec_eval_data.param_values, y=rej_percentage(sec_eval), marker='o')

sp1.xticks(sec_eval.sec_eval_data.param_values)
sp1.xlabel(sec_eval.sec_eval_data.param_name)
sp1.ylabel(CMetricAccuracyReject().class_type.capitalize())
sp1.legend()
sp1.title("Security Evaluation Curve")
sp1.apply_params_sec_eval()

# Rej percentage plot code
sp2.xticks(sec_eval.sec_eval_data.param_values)
sp2.xlabel(sec_eval.sec_eval_data.param_name)
sp2.ylabel("% Reject")
sp2.apply_params_sec_eval()

# Dump to file
fig.savefig("not_evading_wb_"+CLF+"_seval")