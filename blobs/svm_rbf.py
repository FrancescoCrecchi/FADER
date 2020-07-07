from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.figure import CFigure
from secml.ml import CKernelRBF
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.classifiers.sklearn.c_classifier_svm_m import CClassifierSVMM
from secml.ml.peval.metrics import CMetricAccuracy

from toy.gamma_estimation import gamma_estimation

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

    # Create a SVM-RBF classifier
    gamma = gamma_estimation(tr, factor=0.5)
    clf = CClassifierSVMM(kernel=CKernelRBF(gamma=gamma))

    # Fit
    clf.verbose = 2
    clf.fit(tr.X, tr.Y)
    clf.verbose = 0

    # Test performance
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Test set accuracy: {:.2f}".format(acc))

    # Wrap in a CClassifierRejectThreshold
    clf_rej = CClassifierRejectThreshold(clf, 0.)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts)

    # Plot decision boundary
    fig = CFigure()
    fig.sp.plot_ds(tr)
    fig.sp.plot_decision_regions(clf_rej, n_grid_points=100, grid_limits=[(-4.5, 4.5), (-4.5, 4.5)])
    fig.savefig('svm_blobs.png')

    # Dump to disk
    clf_rej.save("svm_blobs")

    print("done?")
