import numpy as np
import torch
from secml.array import CArray
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch, CNormalizerMinMax
from torch import nn, optim

from components.rbf_network import RBFNetwork


class CClassifierPyTorchRBFNetwork(CClassifierPyTorch):

    def __init__(self, model, loss=None, optimizer=None, optimizer_scheduler=None, pretrained=False,
                 pretrained_classes=None, input_shape=None, random_state=None, preprocess=None, softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1, transform_train=None, validation_data=None, track_prototypes=False):
        super().__init__(model, loss, optimizer, optimizer_scheduler, pretrained, pretrained_classes, input_shape,
                         random_state, preprocess, softmax_outputs, epochs, batch_size, n_jobs, transform_train,
                         validation_data)
        # Internals
        self._track_prototypes = track_prototypes

    @property
    def track_prototypes(self):
        return self._track_prototypes

    @track_prototypes.setter
    def track_prototypes(self, value):
        self._track_prototypes = value

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

        # HACK: TRACKING PROTOTYPES
        if self.track_prototypes:
            prototypes = [[p.copy() for p in self.model.prototypes]]

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
                if self.track_prototypes:
                    prototypes.append([p.copy() for p in self.model.prototypes])

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
        if self.track_prototypes:
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

    # HACK: Trying to select 3 prototypes per class
    N_PROTO_PER_CLASS = 3
    n_hiddens = N_PROTO_PER_CLASS * dataset.num_classes

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
    prototypes = CArray.zeros((N_PROTO_PER_CLASS * dataset.num_classes, n_features))
    for i in range(dataset.num_classes):
        xi = dataset.X[dataset.Y == i, :]
        proto = xi[CArray.randsample(xi.shape[0], shape=N_PROTO_PER_CLASS), :]
        prototypes[i*N_PROTO_PER_CLASS:(i+1)*N_PROTO_PER_CLASS, :] = proto
    rbf_net.prototypes = [torch.Tensor(prototypes.tondarray())]

    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rbf_net.parameters())
    clf = CClassifierPyTorchRBFNetwork(rbf_net,
                                       loss=loss,
                                       optimizer=optimizer,
                                       input_shape=(n_feats,),
                                       epochs=30,
                                       batch_size=32,
                                       track_prototypes=True,
                                       random_state=random_state)

    # Fit
    clf.verbose = 1
    clf.fit(dataset.X, dataset.Y)
    clf.verbose = 0

    # Track prototypes
    prototypes = np.array(clf._prototypes).squeeze()   # shape = (n_tracks, n_hiddens, n_feats)
    prototypes = [CArray(prototypes[:, i, :]) for i in range(prototypes.shape[1])]

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
