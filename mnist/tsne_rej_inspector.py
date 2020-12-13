from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold

# GAMMA = 1000
EPS = 4.0
N_TEST = 1000

def load_adv_ds(i):
    seval_data = CSecEval.load('dnn_wb_seval_it_%d.gz' % i).sec_eval_data
    eps_idx = seval_data.param_values.tolist().index(EPS)
    X_adv, Y_rej = seval_data.adv_ds[eps_idx].X, seval_data.Y_pred[eps_idx]
    return CDataset(X_adv, Y_rej)

if __name__ == '__main__':
    random_state = 999

    # 1. Load clf_rej
    clf_rej = CClassifierRejectThreshold.load('tsne_rej.gz')
    ptsne = clf_rej.preprocess

    # Load natural points
    from mnist.fit_dnn import get_datasets
    _, _, ts = get_datasets(random_state)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    nat_ds = ts[ts_idxs, :]

    # 2. Load attack points
    adv_ds = load_adv_ds(0)
    for i in range(1, 3):
        adv_ds = adv_ds.append(load_adv_ds(i))

    # Setup
    preproc = clf_rej.preprocess.copy()
    clf_rej.preprocess = None

    # # For each class
    # fig = CFigure(12, 10)
    # for c in nat_ds.classes:
    #     # 3. Take equal parts of natural and adversarial samples
    #     X_nat_c = nat_ds.X[nat_ds.Y == c, :]
    #     X_adv_c = adv_ds.X[adv_ds.Y == c, :]
    #     k = min(X_nat_c.shape[0], X_adv_c.shape[0])
    #     X_nat_c = X_nat_c[:k, :]
    #     X_adv_c = X_adv_c[:k, :]
    #
    #     # 4. Concatenate together with fake labels
    #     X_c = CArray.concatenate(X_nat_c, X_adv_c, axis=0)
    #     Y_c = CArray.ones(X_c.shape[0]) * c # Natural
    #     Y_c[k:] *= -1 # Adversarial
    #
    #     # 5. Pass through ptSNE to obtain a 2d plot
    #     X_2d = ptsne.forward(X_c)
    #     ds_2d = CDataset(X_2d, Y_c)
    #
    #     fig.sp.plot_ds(ds_2d)

    # Label adversarial labels starting from n_classes
    adv_ds.Y += nat_ds.num_classes

    # Concatenate datasets
    cat_ds = nat_ds.append(adv_ds)

    # Pass through ptSNE to obtain a 2d plot
    X_2d = ptsne.forward(cat_ds.X)
    ds_2d = CDataset(X_2d, cat_ds.Y)

    # 6. Plot
    fig = CFigure(12, 10)
    fig.sp.plot_decision_regions(clf_rej,
                                 grid_limits=[[-130, 130], [-130, 130]],
                                 alpha=0.9,
                                 n_grid_points=100)

    fig.sp.plot_ds(ds_2d)
    fig.savefig('tsne_rej_inspect')

    clf_rej.preprocess = preproc