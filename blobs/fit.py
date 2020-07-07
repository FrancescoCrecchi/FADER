import torch
from torch import nn, optim

from secml.figure import CFigure
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml import CNormalizerMinMax

from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork


if __name__ == '__main__':
    seed = 999
    torch.manual_seed(seed)

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

    # Create a blobs RBF-Net classifier
    n_classes = len(centers)
    model = RBFNetwork(n_features, n_classes, n_classes)
    clf = CClassifierPyTorchRBFNetwork(model,
                                       loss=nn.CrossEntropyLoss(),
                                       optimizer=optim.SGD(model.parameters(), lr=1e-2),
                                       input_shape=(n_features,),
                                       epochs=250,
                                       batch_size=32,
                                       track_prototypes=True,
                                       random_state=seed)

    # Speedup training by prototype init.
    proto = CArray.zeros((n_classes, tr.X.shape[1]))
    for c in range(n_classes):
        proto[c, :] = tr.X[tr.Y == c, :][0, :]
    model.prototypes = [torch.Tensor(proto.tondarray()).float()]

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

    # Plot prototypes
    import numpy as np
    prototypes = np.array(clf_rej.clf._prototypes).squeeze()  # shape = (n_tracks, n_hiddens, n_feats)
    prototypes = [CArray(prototypes[:, i, :]) for i in range(prototypes.shape[1])]
    for proto in prototypes:
        fig.sp.plot_path(proto)

    fig.savefig('rbfnet_blobs')

    # Dump to disk
    clf_rej.save("rbfnet_blobs")

    print("done?")