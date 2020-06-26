import os

from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetricAccuracyReject, CMetricAccuracy

DSET = 'mnist'
EVAL_TYPE = 'wb'
CLFS = ['dnn', 'nr', 'dnr', 'rbf_net'] #,'tsne_rej', 'tnr']
FNAME = 'all_'+EVAL_TYPE+'_seval_sigma_2' # 'tsne_rej_test_gamma'


def compute_performance(seval_data):
    # Compute performance for eps > 0 (i.e. CMetricAccuracyReject)
    perf = CArray.zeros(shape=(seval_data.param_values.size,))
    acc_metric = CMetricAccuracy()
    rej_metric = CMetricAccuracyReject()
    for i, k in enumerate(seval_data.param_values):
        pred = seval_data.Y_pred[i]
        if k == 0:
            # CMetricAccuracy
            metric = acc_metric
        else:
            # CMetricAccuracyReject
            metric = rej_metric
        perf[i] = metric.performance_score(y_true=seval_data.Y, y_pred=pred)
    # TODO: AVERAGE ACROSS MULTIPLE RUNS?

    return perf


# Plot sec_eval with reject percentage
def rej_percentage(seval):
    return [(p == -1).sum() / p.shape[0] for p in seval.sec_eval_data.Y_pred]


if __name__ == '__main__':

    fig = CFigure(height=8, width=10)
    # Sec eval plot code
    sp1 = fig.subplot(2, 1, 1)
    sp2 = fig.subplot(2, 1, 2)
    # This is done here to make 'markevery' work correctly
    for clf in CLFS:
        # Load sec_eval
        if clf == 'dnn':
            seval = CSecEval.load(os.path.join(DSET, clf + '_seval.gz'))
        else:
            seval = CSecEval.load(os.path.join(DSET, clf + '_'+EVAL_TYPE+'_seval.gz'))

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
    out_file = os.path.join(DSET, FNAME)
    print("- Saving output to file: {}".format(out_file))
    fig.savefig(os.path.join(DSET, FNAME))
