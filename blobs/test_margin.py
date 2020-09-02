from secml.figure import CFigure
from secml.ml import CClassifierPyTorch
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml.peval.metrics import CMetricAccuracy

import torch
from torch import nn, optim

from components.rbf_network import CategoricalHingeLoss

if __name__ == '__main__':
    seed = 999
    torch.manual_seed(seed)

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [4, -2], [3, 4]]  # Centers of the clusters
    cluster_std = 0.75  # Standard deviation of the clusters
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=seed).load()

    tr, ts = CTrainTestSplit(test_size=0.3, random_state=seed).split(ds)

    # Create a blobs linear classifier
    n_classes = len(centers)
    model = nn.Linear(n_features, n_classes)
    # loss = nn.CrossEntropyLoss()
    loss = CategoricalHingeLoss()
    clf = CClassifierPyTorch(model,
                             loss=loss,
                             optimizer=optim.SGD(model.parameters(), lr=1e-2),
                             input_shape=(n_features,),
                             epochs=250,
                             batch_size=64,
                             random_state=seed)
    # Fit
    clf.verbose = 2
    clf.fit(tr.X, tr.Y)
    clf.verbose = 0

    # Test performance
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Test set accuracy: {:.2f}".format(acc))

    # Plot decision boundary
    fig = CFigure()
    fig.sp.plot_ds(tr)
    fig.sp.plot_decision_regions(clf, n_grid_points=100, grid_limits=[(-7.5, 7.5), (-7.5, 7.5)])
    fig.show()

    print("done?")