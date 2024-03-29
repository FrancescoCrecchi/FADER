from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject


def run_attack(attack, dset, n_to_plot=10):
    N = dset.Y.size
    eva_y_pred = CArray.zeros((N, ))
    fig = CFigure(height=5 * n_to_plot, width=16)
    c = 0
    for i in range(N):

        print(i)

        x0, y0 = dset[i, :].X, dset[i, :].Y

        # Rerun attack to have '_f_seq' and 'x_seq'
        y, _, _, _ = attack.run(x0, y0)
        eva_y_pred[i] = y.item()

        # To plot?
        if c < n_to_plot and y == y0:   # NOT EVADING
            print("PLOT++")
            # Loss curve
            sp1 = fig.subplot(n_to_plot, 2, c * 2 + 1)
            sp1.plot(attack._f_seq, marker='o', label='PGDExp')
            sp1.grid()
            sp1.xticks(range(attack._f_seq.shape[0]))
            sp1.xlabel('Iteration')
            sp1.ylabel('Loss')
            sp1.legend()

            # Confidence curves
            n_iter, n_classes = attack.x_seq.shape[0], clf.n_classes
            scores = CArray.zeros((n_iter, n_classes))
            for k in range(attack.x_seq.shape[0]):
                scores[k, :] = clf.decision_function(attack.x_seq[k, :])

            sp2 = fig.subplot(n_to_plot, 2, c * 2 + 2)
            for k in range(-1, clf.n_classes - 1):
                sp2.plot(scores[:, k], marker='o', label=str(k))
            sp2.grid()
            sp2.xticks(range(attack.x_seq.shape[0]))
            sp2.xlabel('Iteration')
            sp2.ylabel('Confidence')
            sp2.legend()

            c += 1

    perf = CMetricAccuracyReject().performance_score(y_true=dset.Y, y_pred=eva_y_pred)
    print("Performance under attack: {0:.2f}".format(perf))

    fig.savefig("wb_attack_tuning.png")

CLF = 'svm'
# TYPE = None
if __name__ == '__main__':
    seed = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=seed).load()

    tr, ts = CTrainTestSplit(test_size=0.3, random_state=seed).split(ds)

    # nmz = CNormalizerMinMax()
    # tr.X = nmz.fit_transform(tr.X)
    # ts.X = nmz.transform(ts.X)

    # Load pre-trained clf
    clf = CClassifierRejectThreshold.load('{}_blobs.gz'.format(CLF))
    # clf = CClassifierRejectThreshold.load('{}_blobs_{}.gz'.format(CLF, seed))
    # clf = CClassifierRejectThreshold.load('{}_blobs_{}.gz'.format(CLF, TYPE))
    # Check estimated performance
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Test set accuracy: {:.2f}".format(acc))

    # Create attack
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    dmax = 3.0  # Maximum perturbation
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.1,
        'eta_min': 0.1,
        'eta_max': None,
        'max_iter': 100,
        'eps': 1e-5
    }

    pgd_attack = CAttackEvasionPGDExp(
        classifier=clf,
        # surrogate_classifier=clf,
        double_init_ds=tr,
        distance=noise_type,
        dmax=dmax,
        ub=None,
        lb=None,
        solver_params=solver_params,
        y_target=y_target)

    # Run attack
    pgd_attack.verbose = 2
    run_attack(pgd_attack, ts[:30, :])      # DEBUG: REMOVE THIS!
    pgd_attack.verbose = 0

    # Dump attack to disk
    pgd_attack.save("{}_attack".format(CLF))
    # pgd_attack.save("{}_attack_{}".format(CLF, seed))
    # pgd_attack.save("{}_attack_{}".format(CLF, TYPE))

    print("done?")