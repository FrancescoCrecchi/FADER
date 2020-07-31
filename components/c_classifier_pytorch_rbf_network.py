import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad

from secml.array import CArray
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch, CNormalizerMinMax, CClassifier

from components.rbf_network import RBFNetwork


def grad_norm(loss, inputs):
    bs = inputs.size(0)
    g = grad(loss, inputs, retain_graph=True)[0] * bs
    g = g.view(bs, -1)
    norm2 = g.norm(2, 1).mean()
    return norm2


def gPenalty(inputs, loss, lam, q):
    # Gradient penalty
    # bs, d_in = inputs.size()
    g = grad(loss, inputs, create_graph=True)[0]  # * bs
    qnorms = g.norm(q, 1)
    # lam = lam * math.pow(d_in, 1. - 1. / q)
    return lam * qnorms.mean()  # / 2.


class CClassifierPyTorchRBFNetwork(CClassifierPyTorch):

    def __init__(self, model, loss=None, optimizer=None, optimizer_scheduler=None, pretrained=False,
                 pretrained_classes=None, input_shape=None, random_state=None, preprocess=None, softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1, transform_train=None, validation_data=None, track_prototypes=False,
                 sigma=0.):
        super().__init__(model, loss, optimizer, optimizer_scheduler, pretrained, pretrained_classes, input_shape,
                         random_state, preprocess, softmax_outputs, epochs, batch_size, n_jobs, transform_train,
                         validation_data)
        # Internals
        self._track_prototypes = track_prototypes
        self._sigma = sigma

        self._history = None
        self._prototypes = None

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
                                         num_workers=self.n_jobs - 1,
                                         transform=self._transform_train)  # , shuffle=True)

        if self._validation_data:
            vali_loader = self._data_loader(self._validation_data.X,
                                            self._validation_data.Y,
                                            batch_size=self._batch_size,
                                            num_workers=self.n_jobs - 1)

        if self._history is None:  # FIRST RUN
            tr_loss, vl_loss = [], []
            xentr_loss, gnorm2, reg, weight_decay = [], [], [], []
            # HACK: TRACKING PROTOTYPES
            if self.track_prototypes:
                prototypes = [[p.clone().detach().cpu().numpy() for p in self.model.prototypes]]
        else:
            tr_loss, vl_loss = self._history['tr_loss'], self._history['vl_loss']
            xentr_loss, gnorm2, reg, weight_decay = self._history['xentr_loss'], self._history['grad_norm'], \
                                                    self._history['penalty'], self._history['weight_decay']
            prototypes = self._prototypes

        for epoch in range(self._epochs):
            train_loss = xentr = grad_norm2 = cum_penalty = 0.
            batches = 0
            for data in train_loader:
                batches += 1
                self._optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                # HACK: REGULARIZATION REQUIRES INPUT GRADIENT
                inputs.requires_grad = True
                outputs = self._model(inputs)
                loss = self._loss(outputs, labels)
                # Logging
                xentr += loss.item()
                grad_norm2 += grad_norm(loss, inputs).item()

                # HACK: Gradient norm regularization
                if self._sigma > 0:
                    penalty = gPenalty(inputs, loss, self._sigma, 2)
                    loss += penalty.item()
                    cum_penalty += penalty.item()
                loss.backward()
                self._optimizer.step()
                # Accumulate loss
                train_loss += loss.item()

            # Mean across batches
            train_loss /= (batches + 1)
            xentr /= (batches + 1)
            grad_norm2 /= (batches + 1)
            cum_penalty /= (batches + 1)

            # Linear layer weight norm
            wd = -9999
            # CClassifierRBFNetwork
            try:
                wd = list(self.model.classifier.parameters())[0].norm(2).item()
            except:
                pass
            # CClassifierDeepRBFNetwork
            try:
                wd = 0.
                # Accounting for linear layers of '_layer_clfs' combiners
                for clf in self.model._layer_clfs:
                    wd += list(clf._combiner.parameters())[0].norm(2).item()
                # And for combiner one
                wd += list(self.model._combiner.parameters())[0].norm(2).item()
            except:
                pass

            self.logger.debug(
                "[DEBUG] Epoch {} -> loss: {:.2e} (xentr:{:.3e}, grad_norm2:{:.3e}, penalty:{:.3e}, wd: {:.3e})".
                    format(epoch + 1,
                           train_loss,
                           xentr,
                           grad_norm2,
                           cum_penalty,
                           wd
                           )
            )

            # print statistics
            if epoch % 10 == 0:

                # HACK: TRACKING PROTOTYPES
                if self.track_prototypes:
                    prototypes.append([p.clone().detach().cpu().numpy() for p in self.model.prototypes])

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
                            vali_loss += loss.item()
                        # accumulate
                        vali_loss /= vali_batches
                        # store
                        vl_loss.append(vali_loss)

                    # Logging
                    self.logger.info('[epoch: %d] TR loss: %.3e - VL loss: %.3e' % (epoch + 1, train_loss, vali_loss))
                    self._model.train()  # restore training mode
                else:
                    # Logging
                    self.logger.info('[epoch: %d] TR loss: %.3f' % (epoch + 1, train_loss))

                # Update curves
                tr_loss.append(train_loss)
                xentr_loss.append(xentr)
                gnorm2.append(grad_norm2)
                reg.append(cum_penalty)
                weight_decay.append(wd)

            if self._optimizer_scheduler is not None:
                self._optimizer_scheduler.step()

        self._trained = True

        # HACK: STORING PROTOTYPES
        if self.track_prototypes:
            self._prototypes = prototypes

        # HACK: Store training data for plots
        self._history = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'xentr_loss': xentr_loss,
            'grad_norm': gnorm2,
            'penalty': reg,
            'weight_decay': weight_decay
        }

        return self._model


def plot_decision_function(sp, clf, c, plot_background=True, levels=None,
                           grid_limits=None, n_grid_points=30, cmap=None, colorbar=True):
    if not isinstance(clf, CClassifier):
        raise TypeError("'clf' must be an instance of `CClassifier`.")

    if cmap is None:
        if clf.n_classes <= 6:
            colors = ['blue', 'red', 'lightgreen', 'black', 'gray', 'cyan']
            cmap = colors[:clf.n_classes]
        else:
            cmap = 'jet'

    if levels is None:
        levels = CArray.arange(0.5, clf.n_classes).tolist()

    # DEBUG: FC wrong class in plots!

    # f = None
    # if issubclass(type(clf), CClassifierReject):
    #     def fun_rej(x):
    #         y_pred = clf.predict(x)
    #         # Restore as 'n+1' the rejection class
    #         y_pred[y_pred == -1] = clf.n_classes - 1
    #         return y_pred
    #
    #     f = fun_rej
    # else:
    #     f = clf.predict
    f = lambda x: clf.decision_function(x)[:, c]

    sp.plot_fun(func=f,  # clf.predict,
                multipoint=True,
                colorbar=colorbar,
                n_colors=clf.n_classes,
                cmap=cmap,
                levels=levels,
                plot_background=plot_background,
                grid_limits=grid_limits,
                n_grid_points=n_grid_points,
                alpha=0.5)

    sp.apply_params_clf()


SIGMA = 1.0
EPOCHS = 300
BATCH_SIZE = 32
if __name__ == '__main__':
    random_state = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-4, 0], [4, -4], [4, 4]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters

    from secml.data.loader import CDLRandomBlobs

    dataset = CDLRandomBlobs(n_features=n_features,
                             centers=centers,
                             cluster_std=cluster_std,
                             n_samples=n_samples,
                             random_state=random_state).load()

    # Select 30K samples to train DNN
    from secml.data.splitter import CTrainTestSplit

    tr, ts = CTrainTestSplit(train_size=1000, random_state=random_state).split(dataset)

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    ts.X = nmz.transform(ts.X)

    n_feats = dataset.X.shape[1]

    # HACK: Trying to select 3 prototypes per class
    N_PROTO_PER_CLASS = 5
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
    # # HACK: FIXING BETA
    # rbf_net.betas = 1.0
    # rbf_net.train_betas = False
    # HACK: SETTING PROTOTYPES
    prototypes = CArray.zeros((N_PROTO_PER_CLASS * dataset.num_classes, n_features))
    for i in range(tr.num_classes):
        xi = tr.X[tr.Y == i, :]
        proto = xi[CArray.randsample(xi.shape[0], shape=N_PROTO_PER_CLASS), :]
        prototypes[i * N_PROTO_PER_CLASS:(i + 1) * N_PROTO_PER_CLASS, :] = proto
    rbf_net.prototypes = [torch.Tensor(prototypes.tondarray())]

    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rbf_net.parameters())
    clf = CClassifierPyTorchRBFNetwork(rbf_net,
                                       loss=loss,
                                       optimizer=optimizer,
                                       input_shape=(n_feats,),
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       track_prototypes=True,
                                       sigma=SIGMA,
                                       random_state=random_state)

    # Fit
    clf.verbose = 2
    clf.fit(tr.X, tr.Y)
    clf.verbose = 0

    # Plot training curves
    from mnist.rbf_net import plot_train_curves

    fig = plot_train_curves(clf._history, SIGMA)
    fig.savefig("c_classifier_rbf_network_curves_SIGMA_{:.3e}.png".format(SIGMA))

    # Track prototypes
    prototypes = np.array(clf._prototypes).squeeze()  # shape = (n_tracks, n_hiddens, n_feats)
    prototypes = [CArray(prototypes[:, i, :]) for i in range(prototypes.shape[1])]

    # Wrap in a CClassifierRejectThreshold
    from secml.ml.classifiers.reject import CClassifierRejectThreshold

    clf_rej = CClassifierRejectThreshold(clf, 0.)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts)

    # Plot decision regions
    fig = CFigure()
    fig.sp.plot_ds(tr)
    fig.sp.plot_decision_regions(clf_rej, n_grid_points=100, grid_limits=[(-1, 2), (-1, 2)])
    # Plot prototypes
    for proto in prototypes:
        fig.sp.plot_path(proto)
    fig.title('RBFNetwork Classifier - Sigma: {:.3f}'.format(SIGMA))
    fig.savefig('c_classifier_rbf_network_SIGMA_{:.3e}_decision_regions.png'.format(SIGMA))

    # Test plot
    fig = CFigure(5, 21)
    fig.subplot(1, clf_rej.n_classes)
    for i in range(clf_rej.n_classes):
        sp = fig.subplot(1, clf_rej.n_classes, i)
        sp.title("Class %d" % (i - 1))
        sp.plot_ds(tr)
        sp.plot_decision_regions(clf, n_grid_points=100, grid_limits=[(-1.5, 2.5), (-1, 2.5)], plot_background=False)
        plot_decision_function(sp, clf_rej, i - 1, n_grid_points=100, grid_limits=[(-1.5, 2.5), (-1.5, 2.5)],
                               cmap='jet')
        # Plot prototypes
        for proto in prototypes:
            fig.sp.plot_path(proto)
    fig.title('RBFNetwork Classifier - Sigma: {:.3f}'.format(SIGMA))
    # fig.show()
    fig.savefig('c_classifier_rbf_network_SIGMA_{:.3e}_decision_functions.png'.format(SIGMA))

    print('done?')
