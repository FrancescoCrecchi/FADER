import copy

import numpy as np
import torch
from secml.array import CArray
from torch import nn, optim
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch, CNormalizerMinMax

from components.rbf_network import RBFNetwork


class CClassifierPytorchRBFNetwork(CClassifierPyTorch):

    def _fit(self, x, y):
        """Fit PyTorch model.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        """
        if any([self._optimizer is None,
                self._loss is None]):
            raise ValueError("Optimizer and loss should both be defined "
                             "in order to fit the model.")

        train_loader = self._data_loader(x, y, batch_size=self._batch_size,
                                         num_workers=self.n_jobs - 1, transform=self._transform_train)

        if self._validation_data:
            vali_loader = self._data_loader(self._validation_data.X,
                                            self._validation_data.Y,
                                            batch_size=self._batch_size,
                                            num_workers=self.n_jobs - 1)
        prototypes = [copy.deepcopy(self.model.prototypes)]
        for epoch in range(self._epochs):
            train_loss = 0.0
            batches = 0
            for data in train_loader:
                batches += 1
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optimizer.step()
                # accumulate (Simple Moving Average)
                train_loss += (1 / batches) * (loss.item() - train_loss)

            # print statistics
            if epoch % 10 == 0:

                # HACK: TRACKING PROTOTYPES
                prototypes.append(copy.deepcopy(self.model.prototypes))

                if self._validation_data is not None:
                    # Compute validation performance
                    self._model.eval()  # enter evaluation mode
                    with torch.no_grad():
                        vali_loss = 0.0
                        vali_batches = 0
                        for data in vali_loader:
                            vali_batches += 1
                            inputs, labels = data
                            inputs = inputs.to(self._device)
                            labels = labels.to(self._device)
                            outputs = self._model(inputs)
                            loss = self._loss(outputs, labels)
                            # accumulate
                            vali_loss += (1 / vali_batches) * (loss.item() - vali_loss)
                    # Logging
                    self.logger.info('[epoch: %d] TR loss: %.3e - VL loss: %.3e' % (epoch + 1, train_loss, vali_loss))
                    self._model.train()  # restore training mode
                else:
                    # Logging
                    self.logger.info('[epoch: %d] TR loss: %.3f' % (epoch + 1, train_loss))

            if self._optimizer_scheduler is not None:
                self._optimizer_scheduler.step()

        self._trained = True
        # HACK: STORING PROTOTYPES
        self._prototypes = prototypes
        return self._model


if __name__ == '__main__':
    random_state = 999

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
    n_hiddens = 3
    n_classes = dataset.num_classes

    # Torch fix random seed
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # RBFNetwork
    rbf_net = RBFNetwork(n_feats, n_hiddens, n_classes)
    # HACK: FIXING BETA
    rbf_net.betas = 10
    rbf_net.train_betas = False
    # HACK: SELECT ONE PROTOTYPE PER CLASS
    prototypes = CArray.zeros((dataset.num_classes, n_features))
    for i in range(dataset.num_classes):
        xi = dataset.X[dataset.Y == i, :]
        proto = xi[CArray.randsample(xi.shape[0], shape=1).item(), :]
        prototypes[i, :] = proto
    rbf_net.prototypes = [torch.Tensor(prototypes.tondarray())]

    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rbf_net.parameters())
    clf = CClassifierPytorchRBFNetwork(rbf_net,
                                       loss=loss,
                                       optimizer=optimizer,
                                       input_shape=(n_feats,),
                                       epochs=300,
                                       batch_size=32,
                                       random_state=random_state)

    # Fit
    clf.verbose = 1
    clf.fit(dataset.X, dataset.Y)
    clf.verbose = 0

    # Track prototypes
    prototypes = np.dstack([proto[0].detach().numpy() for proto in clf._prototypes]).T   # shape = (n_tracks, n_hiddens, n_feats)
    prototypes = [CArray(prototypes[:, :, i]) for i in range(prototypes.shape[2])]

    # Test plot
    fig = CFigure()
    fig.sp.plot_ds(dataset)
    fig.sp.plot_decision_regions(clf, n_grid_points=100, grid_limits=[(-0.25, 1), (-0.25, 1)] )
    # Plot prototypes
    for proto in prototypes:
        fig.sp.plot_path(proto)
    fig.title('RBFNetwork Classifier')
    fig.show()
    # fig.savefig('c_classifier_rbf_network.png')

    print('done?')
