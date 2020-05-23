from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracyReject, CMetricAccuracy

# Load sec_eval
seval = CSecEval.load('clf_rej_bb_seval.gz')

# Load clf_rej
clf_rej = CClassifierRejectThreshold.load('clf_rej.gz')

# Compute performance at eps=0 (i.e. CMetricAccuracy)
pred = clf_rej.predict(seval.sec_eval_data.adv_ds[0].X)
acc_at_zero = CMetricAccuracy().performance_score(seval.sec_eval_data.Y, pred)

# Compute performance for eps > 0 (i.e. CMetricAccuracyReject)
perf = CArray.zeros(shape=(seval.sec_eval_data.param_values.size,))
perf[0] = acc_at_zero
metric = CMetricAccuracyReject()
for k in range(1, seval.sec_eval_data.param_values.size):
    pred, scores = clf_rej.predict(seval.sec_eval_data.adv_ds[k].X, return_decision_function=True)
    perf[k] = metric.performance_score(y_true=seval.sec_eval_data.Y, y_pred=pred, score=scores)
# TODO: AVERAGE ACROSS MULTIPLE RUNS?

# Plot sec_eval with reject percentage
rej_percentage = lambda seval: [(p == -1).sum()/p.shape[0] for p in seval.sec_eval_data.Y_pred]

fig = CFigure(height=8, width=10)
# Sec eval plot code
sp1 = fig.subplot(2,1,1)
# This is done here to make 'markevery' work correctly
sp1.xticks(seval.sec_eval_data.param_values)
sp1.plot(seval.sec_eval_data.param_values, perf, label='clf_rej',
          linestyle='-', color=None, marker='o',
          markevery=sp1.get_xticks_idx(seval.sec_eval_data.param_values))
sp1.xlabel(seval.sec_eval_data.param_name)
sp1.ylabel(metric.class_type.capitalize())
sp1.legend()
sp1.title("Security Evaluation Curve")
sp1.apply_params_sec_eval()
# Rej percentage plot code
sp2 = fig.subplot(2, 1, 2)
sp2.plot(seval.sec_eval_data.param_values, y=rej_percentage(seval), marker='o')
sp2.xticks(seval.sec_eval_data.param_values)
sp2.xlabel(seval.sec_eval_data.param_name)
sp2.ylabel("% Reject")
sp2.apply_params_sec_eval()

# Dump to file
fig.savefig('clf_rej_bb_seval')