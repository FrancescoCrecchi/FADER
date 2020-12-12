from secml.adv.seceval import CSecEvalData
from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold

# GAMMA = 1000
EPS = 3.0

if __name__ == '__main__':
    # 1. Load clf_rej
    # TSNE -> MINMAX -> KDE
    # clf_rej = CClassifierRejectThreshold.load('tsne_rej_test_gamma_{:}.gz'.format(GAMMA))
    # clf_rej = CClassifierRejectThreshold.load('/home/crecchi/DNR/mnist/tsne_rej/tsne_rej_test_gamma/tsne_rej_test_gamma_10000.0.gz')
    # clf_rej = CClassifierRejectThreshold.load('tsne_rej.gz')
    ptsne = clf_rej.preprocess

    # # TSNE -> SVM
    # clf_rej = CClassifierRejectThreshold.load('tsne_nr.gz')
    # ptsne = clf_rej.clf.preprocess.preprocess

    # 2. Load attack points
    seval_data = CSecEvalData.load('tsne_rej_bb_seval_it_0.gz')

    # Fwd through CReducerTSNE
    eps_idx = seval_data.param_values.tolist().index(EPS)
    X_adv, Y_rej = seval_data.adv_ds[eps_idx].X, seval_data.Y_pred[eps_idx]
    X_2d = ptsne.forward(X_adv)
    adv_ds_2d = CDataset(X_2d, Y_rej)

    # # TRAINING DATA
    # from mnist.fit_dnn import get_datasets
    # random_state = 999
    # N_TRAIN, N_TEST = 10000, 1000
    # _, vl, _ = get_datasets(random_state)
    # tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    # tr_sample = vl[tr_idxs, :]
    # X_2d = ptsne.forward(tr_sample.X)
    # adv_ds_2d = CDataset(X_2d, tr_sample.Y)

    # Plot points together with decision regions
    fig = CFigure(12, 10)

    # HACK: Detach from preprocess to plot
    preproc = clf_rej.preprocess.copy()
    clf_rej.preprocess = None
    # preproc = clf_rej.clf.preprocess.preprocess.copy()
    # clf_rej.clf.preprocess.preprocess = None

    fig.sp.plot_decision_regions(clf_rej,
                                 grid_limits=[[-130, 130], [-130, 130]],
                                 # levels=CArray.arange(0.5, clf_rej.n_classes).tolist(),
                                 alpha=1.0,
                                 # colorbar=True,
                                 n_grid_points=100)
    fig.sp.plot_ds(adv_ds_2d)

    clf_rej.preprocess = preproc
    # clf_rej.clf.preprocess.preprocess = preproc

    # Dump to disk
    fig.savefig('tsne_rej_visualize')
    # fig.savefig('tsne_nr_visualize')
