import torch
from torch import nn, optim
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch, CNormalizerMinMax

from components.rbf_network import RBFNetwork

if __name__ == '__main__':
    random_state = None #999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters

    from secml.data.loader import CDLRandomBlobs

    dataset = CDLRandomBlobs(n_features=n_features,
                             centers=centers,
                             cluster_std=cluster_std,
                             n_samples=n_samples,
                             random_state=random_state).load()

    nmz = CNormalizerMinMax()
    dataset.X = nmz.fit_transform(dataset.X)

    n_feats = dataset.X.shape[1]
    n_hiddens = 10
    n_classes = dataset.num_classes

    # Torch fix random seed
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # RBFNetwork
    rbf_net = RBFNetwork(n_feats, n_hiddens, n_classes, beta=100.0)
    # print(list(rbf_net.parameters())[0][:10])
    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rbf_net.parameters())
    clf = CClassifierPyTorch(rbf_net,
                             loss=loss,
                             optimizer=optimizer,
                             input_shape=(n_feats, ),
                             epochs=500,
                             batch_size=32,
                             random_state=random_state)

    # Fit
    clf.verbose = 1
    clf.fit(dataset.X, dataset.Y)
    clf.verbose = 0

    # Test plot
    fig = CFigure()
    fig.sp.plot_ds(dataset)
    fig.sp.plot_decision_regions(clf, n_grid_points=100)
    fig.title('RBFNetwork Classifier')
    fig.show()
    # fig.savefig('c_classifier_rbf_network.png')

    print('done?')
