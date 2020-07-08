import torch
from torch import nn, optim
from secml.array import CArray
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch, CClassifier, CNormalizerDNN
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


def rbf_network(dnn, layers, n_hiddens=100,
                epochs=300, batch_size=32,
                validation_data=None, sigma=0.0,
                track_prototypes=False, random_state=None):
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    # RBFNetwork
    model = RBFNetwork(n_feats, n_hiddens, dnn.n_classes)
    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
    # HACK: TRACKING PROTOTYPES
    return CClassifierPyTorchRBFNetwork(model,
                                        loss=loss,
                                        optimizer=optimizer,
                                        input_shape=(sum(n_feats, )),
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=validation_data,
                                        track_prototypes=track_prototypes,
                                        sigma=sigma,
                                        random_state=random_state)


class CClassifierRBFNetwork(CClassifier):

    def __init__(self, dnn, layers, n_hiddens=100,
                 epochs=300, batch_size=32,
                 validation_data=None,
                 sigma=0.0,  # DEFAULT: No regularization!
                 track_prototypes=False,
                 random_state=None):

        # Param checking
        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens] * len(layers)
        self._n_hiddens = n_hiddens

        # RBF Network
        self._clf = rbf_network(dnn, layers, n_hiddens, epochs, batch_size, validation_data, sigma, track_prototypes, random_state)
        super(CClassifierRBFNetwork, self).__init__()

        self._layers = layers
        self._features_extractors = {}
        for layer in self._layers:
            self._features_extractors[layer] = CNormalizerDNN(dnn, layer)

        # Utils
        self._num_features = CArray([CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers])

    @property
    def verbose(self):
        return self._clf.verbose

    @verbose.setter
    def verbose(self, value):
        self._clf.verbose = value

    def _create_scores_dataset(self, x):
        caching = self._cached_x is not None

        # Compute layer representations
        concat_scores = CArray.zeros(shape=(x.shape[0], self._num_features.sum()))
        start = 0
        for i, l in enumerate(self._layers):
            out = self._features_extractors[l].forward(x, caching=caching)
            out = out.reshape((x.shape[0], -1))
            concat_scores[:, start:start + self._num_features[i].item()] = out
            start += self._num_features[i].item()

        return concat_scores

    def _fit(self, x, y):
        x = self._create_scores_dataset(x)
        if self._clf._validation_data:
            self._clf._validation_data.X = self._create_scores_dataset(self._clf._validation_data.X)
        self._clf.fit(x, y)
        return self

    def _forward(self, x):
        caching = self._cached_x is not None
        # Compute internal layer repr.
        x = self._create_scores_dataset(x)
        # Pass through RBF network
        scores = self._clf.forward(x, caching=caching)
        return scores

    def _backward(self, w):
        grad = CArray.zeros(self.n_features)
        # RBF Network Gradient
        grad_combiner = self._clf.backward(w)
        # Propagate through NormalizerDNN(s) and accumulate
        start = 0
        for i, l in enumerate(self._layers):
            # backward pass to layer clfs of their respective w
            grad += self._features_extractors[l].backward(
                w=grad_combiner[:, start:start + self._num_features[i].item()])
            start += self._num_features[i].item()

        return grad

    @property
    def prototypes(self):
        res = [CArray(proto.clone().detach().cpu().numpy()) for proto in self._clf.prototypes]
        return res

    @prototypes.setter
    def prototypes(self, x):
        proto_feats = []
        for i, l in enumerate(self._layers):
            f_x = self._features_extractors[l].transform(x[:self._n_hiddens[i], :])
            proto_feats.append(torch.Tensor(f_x.tondarray()).to(self._clf._device))
        self._clf.model.prototypes = proto_feats

    @property
    def history(self):
        return self._clf._history

    @property
    def _grad_requires_forward(self):
        return True

    # TODO: Expose Betas


class CClassifierRejectRBFNet(CClassifierRejectThreshold):

    @property
    def _grad_requires_forward(self):
        return True


# PARAMETERS
SIGMA = 0.01
EPOCHS = 250
BATCH_SIZE = 128


def plot_train_curves(history, sigma):
    fig = CFigure()
    fig.sp.plot(history['tr_loss'], label='TR', marker="o")
    fig.sp.plot(history['vl_loss'], label='VL', marker="o")
    fig.sp.plot(history['xentr_loss'], label='xentr', marker="o")
    fig.sp.plot(history['grad_norm'], label='g_norm2', marker="o")
    # fig.sp.plot(history['weight_decay'], label='decay', marker="o")
    fig.sp.plot(history['penalty'], label='penalty', marker="o")
    fig.sp.title("Training Curves - Sigma: {}".format(sigma))
    fig.sp.legend()
    fig.sp.grid()
    return fig


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    # HACK: SELECTING VALIDATION DATA (shape=2*N_TEST)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=2*N_TEST, random_state=random_state)
    vl_sample = ts[ts_idxs[:N_TEST], :]
    ts_sample = ts[ts_idxs[N_TEST:], :]

    # Create DNR
    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    n_hiddens = [250, 250, 50]
    rbf_net = CClassifierRBFNetwork(dnn, layers,
                                    n_hiddens=n_hiddens,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=vl_sample,
                                    sigma=SIGMA,              # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                    random_state=random_state)

    # Initialize prototypes with some training samples
    h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    proto = tr_sample.X[idxs, :]
    rbf_net.prototypes = proto

    # Fit DNR
    rbf_net.verbose = 2  # DEBUG
    rbf_net.fit(tr_sample.X, tr_sample.Y)
    rbf_net.verbose = 0

    # Plot training curves
    fig = plot_train_curves(rbf_net.history, SIGMA)
    fig.savefig("rbf_net_train_sigma_{:.3f}_curves.png".format(SIGMA))

    # Check test performance
    y_pred = rbf_net.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("RBFNet Accuracy: {}".format(acc))

    # We can now create a classifier with reject
    clf_rej = CClassifierRejectRBFNet(rbf_net, 0.)

    # Set threshold (FPR: 10%)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Dump to disk
    clf_rej.save('rbf_net_sigma_{:.3f}_{}'.format(SIGMA, EPOCHS))