import os
import numpy as np
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetricAccuracyReject, CMetricAccuracy

DSET = 'cifar10'
EVAL_TYPE = 'bb'
# CLFS = ['rbf_net_sigma_{:.1f}'.format(sigma) for sigma in np.arange(4, dtype=float)]
# CLFS = ['dnn', 'nr', 'dnr'] + CLFS #, 'deep_rbf_net_sigma_1'] #,'tsne_rej', 'tnr']
# CLFS = ['dnn', 'nr', 'dnr', 'rbf_net_sigma_0.000_30', 'rbf_net_sigma_0.000_250', 'rbf_net_sigma_0.010_250']
# CLFS = ['dnn', 'nr', 'dnr', 'rbf_net_sigma_0.000_250',  'deep_rbf_net_train_sigma_0.000_250']
CLFS = ['nr', 'dnr',
        # 'rbf_net_sigma_0.000_250', 'rbf_net_sigma_0.000_500_1xclass',
        # 'rbf_net_sigma_0.000_250_4x', 'rbf_net_sigma_0.010_1000']
        # 'rbf_net_sigma_0.000_250_nr_like',
        'deep_rbf_net_train_sigma_0.000_250'
        ]
# CLFS = ['dnn', 'adv_reg_cnn']
N_ITER = 3
FNAME = 'all_'+EVAL_TYPE+'_deep_rbf_net' #'_rbf_net_nr_like' # 'tsne_rej_test_gamma'
# FNAME = 'adv_reg_dnn'


# Plot sec_eval with reject percentage
def rej_percentage(sec_eval_data):
    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):
        perf[k] = (sec_eval_data.Y_pred[k] == -1).sum() / sec_eval_data.Y_pred[k].shape[0]
    return perf


def acc_rej_performance(sec_eval_data):
    # Compute performance for eps > 0 (i.e. CMetricAccuracyReject)
    acc_metric = CMetricAccuracy()
    rej_metric = CMetricAccuracyReject()

    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):
        pred = sec_eval_data.Y_pred[k]
        if k == 0:
            # CMetricAccuracy
            metric = acc_metric
        else:
            # CMetricAccuracyReject
            metric = rej_metric
        perf[k] = metric.performance_score(y_true=sec_eval_data.Y, y_pred=pred)

    return perf


def compute_performance(sec_eval_data, perf_eval_fun):

    if not isinstance(sec_eval_data, list):
        sec_eval_data = [sec_eval_data]

    n_sec_eval = len(sec_eval_data)
    n_param_val = sec_eval_data[0].param_values.size
    perf = CArray.zeros((n_sec_eval, n_param_val))

    # Single runs curves
    for i in range(n_sec_eval):
        if sec_eval_data[i].param_values.size != n_param_val:
            raise ValueError("the number of sec eval parameters changed!")

        perf[i, :] = perf_eval_fun(sec_eval_data[i])

    # Compute mean and std
    perf_std = perf.std(axis=0, keepdims=False)
    perf_mean = perf.mean(axis=0, keepdims=False)
    # perf = perf.ravel()

    return perf_mean, perf_std


if __name__ == '__main__':

    fig = CFigure(height=8, width=10)
    # Sec eval plot code
    sp1 = fig.subplot(2, 1, 1)
    sp2 = fig.subplot(2, 1, 2)
    # This is done here to make 'markevery' work correctly

    for clf in CLFS:
        if EVAL_TYPE == 'bb':
            # BB SETTINGS
            sec_evals = [CSecEval.load(os.path.join(DSET, clf + '_'+EVAL_TYPE+'_seval.gz'))]
        elif EVAL_TYPE == 'wb':
            # WB SETTINGS
            sec_evals = []
            for it in range(N_ITER):
                # Load sec_eval
                fname = os.path.join(DSET, clf + '_'+EVAL_TYPE+'_seval_it_'+str(it)+'.gz')
                if os.path.exists(fname):
                    seval = CSecEval.load(fname)
                    sec_evals.append(seval)
        else:
            raise ValueError("Unknown EVAL_TYPE!")

        print(" - Plotting ", clf)

        sec_evals_data = [seval.sec_eval_data for seval in sec_evals]

        # Plot performance
        perf, perf_std = compute_performance(sec_evals_data, acc_rej_performance)
        sp1.plot(sec_evals[0].sec_eval_data.param_values, perf, label=clf)
        # Plot mean and std
        std_up = perf + perf_std
        std_down = perf - perf_std
        std_down[std_down < 0.0] = 0.0
        std_down[std_up > 1.0] = 1.0
        sp1.fill_between(sec_evals[0].sec_eval_data.param_values, std_up, std_down, interpolate=False, alpha=0.2)


        # Plot reject percentage
        perf, perf_std = compute_performance(sec_evals_data, rej_percentage)
        sp2.plot(sec_evals[0].sec_eval_data.param_values, perf, label=clf)
        # Plot mean and std
        std_up = perf + perf_std
        std_down = perf - perf_std
        std_down[std_down < 0.0] = 0.0
        std_down[std_up > 1.0] = 1.0
        sp2.fill_between(sec_evals[0].sec_eval_data.param_values, std_up, std_down, interpolate=False, alpha=0.2)

    sp1.xticks(sec_evals[0].sec_eval_data.param_values)
    sp1.xlabel(sec_evals[0].sec_eval_data.param_name)
    sp1.ylabel(CMetricAccuracyReject().class_type.capitalize())
    sp1.legend()
    sp1.title("Security Evaluation Curve")
    sp1.apply_params_sec_eval()

    # Rej percentage plot code
    sp2.xticks(sec_evals[0].sec_eval_data.param_values)
    sp2.xlabel(sec_evals[0].sec_eval_data.param_name)
    sp2.ylabel("% Reject")
    sp2.apply_params_sec_eval()

    # Dump to file
    out_file = os.path.join(DSET, FNAME)
    print("- Saving output to file: {}".format(out_file))
    fig.savefig(os.path.join(DSET, FNAME))
