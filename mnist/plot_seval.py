from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracyReject, CMetricAccuracy

CLFS = ['dnn', 'nr', 'dnr'] #, 'tsne_rej', 'tnr']
FNAME = 'all_wb_seval' # 'tsne_rej_test_gamma'
# from mnist.transferability import CLFS


def compute_performance(seval_data):
    # Compute performance at eps=0 (i.e. CMetricAccuracy)
    pred = seval_data.Y_pred[0]
    acc_at_zero = CMetricAccuracy().performance_score(seval_data.Y, pred)

    # Compute performance for eps > 0 (i.e. CMetricAccuracyReject)
    perf = CArray.zeros(shape=(seval_data.param_values.size,))
    perf[0] = acc_at_zero
    metric = CMetricAccuracyReject()
    for k in range(1, seval_data.param_values.size):
        pred = seval_data.Y_pred[k]
        perf[k] = metric.performance_score(y_true=seval_data.Y, y_pred=pred)
    # TODO: AVERAGE ACROSS MULTIPLE RUNS?

    return perf


# Plot sec_eval with reject percentage
rej_percentage = lambda seval: [(p == -1).sum() / p.shape[0] for p in seval.sec_eval_data.Y_pred]


fig = CFigure(height=8, width=10)
# Sec eval plot code
sp1 = fig.subplot(2, 1, 1)
sp2 = fig.subplot(2, 1, 2)
# This is done here to make 'markevery' work correctly
for clf in CLFS:
    # Load sec_eval
    if clf == 'dnn':
        seval = CSecEval.load(clf + '_seval.gz')
    else:
        seval = CSecEval.load(clf + '_wb_seval.gz')

    print(" - Plotting ", clf)

    # Plot performance
    sp1.plot(seval.sec_eval_data.param_values, compute_performance(seval.sec_eval_data), label=clf,
             linestyle='-', color=None, marker='o',
             markevery=sp1.get_xticks_idx(seval.sec_eval_data.param_values))
    # Plot reject percentage
    sp2.plot(seval.sec_eval_data.param_values, y=rej_percentage(seval), marker='o')

sp1.xticks(seval.sec_eval_data.param_values)
sp1.xlabel(seval.sec_eval_data.param_name)
sp1.ylabel(CMetricAccuracyReject().class_type.capitalize())
sp1.legend()
sp1.title("Security Evaluation Curve")
sp1.apply_params_sec_eval()

# Rej percentage plot code
sp2.xticks(seval.sec_eval_data.param_values)
sp2.xlabel(seval.sec_eval_data.param_name)
sp2.ylabel("% Reject")
sp2.apply_params_sec_eval()

# Dump to file
fig.savefig(FNAME)
