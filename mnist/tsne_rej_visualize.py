import re
from secml.adv.seceval import CSecEval
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold

GAMMA = 1E5
EPS = 3.0


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


if __name__ == '__main__':
    # 1. Load clf_rej
    clf_rej = CClassifierRejectThreshold.load('tsne_rej_test_gamma_{:}.gz'.format(GAMMA))

    # 2. Load attack points
    seval = CSecEval.load('tsne_rej_test_gamma_{:}_bb_seval.gz'.format(GAMMA))

    # Fwd through CReducerTSNE
    eps_idx = seval.sec_eval_data.param_values.tolist().index(EPS)
    X_adv, Y_rej = seval.sec_eval_data.adv_ds[eps_idx].X, seval.sec_eval_data.Y_pred[eps_idx]
    # Obtain 2d point
    X_2d = clf_rej.preprocess.preprocess.forward(X_adv)  # TODO: assign names to preprocesses!
    adv_ds_2d = CDataset(X_2d, Y_rej)

    # Plot points together with decision regions
    fig = CFigure(12, 10)

    # HACK: Detach from preprocesses
    clf_rej.preprocess.preprocess = None
    clf_rej.clf._n_features = 2

    fig.sp.plot_decision_regions(clf_rej, grid_limits=[[-120, 120], [-120, 120]])
    fig.sp.show_legend = True       # TODO: enhance this!
    fig.sp.plot_ds(adv_ds_2d)

    # Dump to disk
    fname = get_valid_filename('tsne_rej_gamma_{:}_eps_{:}_bb_seval.png'.format(GAMMA, EPS))
    fig.savefig(fname)
