import torch
from secml.array import CArray
from secml.figure import CFigure
from secml.ml import CClassifier, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy
from torch import nn, optim

from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.deep_rbf_net import DeepRBFNetwork
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets
from mnist.rbf_net import CClassifierRejectRBFNet, plot_train_curves


def deep_rbf_network(dnn, layers, n_hiddens=100,
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
    model = DeepRBFNetwork(n_feats, n_hiddens, dnn.n_classes)
    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
    # HACK: TRACKING PROTOTYPES
    return CClassifierPyTorchRBFNetwork(model,
                                        loss=loss,
                                        optimizer=optimizer,
                                        input_shape=(sum(n_feats), ),
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=validation_data,
                                        track_prototypes=track_prototypes,
                                        sigma=sigma,
                                        random_state=random_state)


class CClassifierDeepRBFNetwork(CClassifier):

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
        self._clf = deep_rbf_network(dnn, layers, n_hiddens, epochs, batch_size, validation_data, sigma, track_prototypes, random_state)
        super(CClassifierDeepRBFNetwork, self).__init__()

        self._layers = layers
        self._features_extractors = {}
        for layer in self._layers:
            self._features_extractors[layer] = CNormalizerDNN(dnn, layer)

        # Utils
        self.input_shape = dnn.input_shape
        self._classes = dnn.classes
        self._num_features = CArray([CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
                                    + [self.n_classes * len(self._layers)])
        self._device = self._clf._device


    @property
    def verbose(self):
        return self._clf.verbose

    @verbose.setter
    def verbose(self, value):
        self._clf.verbose = value

    def _create_scores_dataset(self, x):
        caching = self._cached_x is not None

        # Compute layer representations
        concat_scores = CArray.zeros(shape=(x.shape[0], self._num_features[:-1].sum()))
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
        return self._clf.model.prototypes

    @prototypes.setter
    def prototypes(self, dset):
        '''
        Set 'DeepRBFNetOnDNN' layer_clfs and combiner prototypes (same Xs for all)
        :param value: Input prototypes vectors
        '''
        proto = []

        # Unpack and reshape
        x, y = dset.X, dset.Y

        f_x = self._create_scores_dataset(x)
        f_x = torch.Tensor(f_x.tondarray()).float().to(self._device)
        # Unpack to 'layer_clfs'
        start = 0
        for i in range(len(self._layers)):
            proto.append(f_x[:self._n_hiddens[i], start:start+self._num_features[i].item()])
            start += self._num_features[i].item()

        # Select one sample per class to init. combiner prototypes
        n_comb_units_per_class = self._n_hiddens[-1]//self.n_classes
        comb_x = CArray.zeros((self.n_classes * n_comb_units_per_class, x.shape[1]))
        start = 0
        for c in range(self.n_classes):
            # Selecting the fist ones for each class, for simplicity
            comb_x[start:start+n_comb_units_per_class, :] = x[y == c, :][:n_comb_units_per_class, :]
            start += n_comb_units_per_class

        # Run dnn on them
        f_x = self._create_scores_dataset(comb_x)
        f_x = torch.Tensor(f_x.tondarray()).float().to(self._device)
        n_samples = f_x.shape[0]

        # Pack activations
        fx = []
        start = 0
        for i in range(len(self._layers)):
            out = self._clf.model._layer_clfs[i](f_x[:, start:start+self._num_features[i].item()].view(n_samples, -1))
            start += self._num_features[i].item()
            fx.append(out)
        # Concatenate into a tensor - shape: (n_samples, n_classes * n_layers)
        fx = torch.cat(fx, 1)
        proto.append(fx)

        # Apply
        self._clf.model.prototypes = proto

    @property
    def betas(self):
        res = [CArray(b) for b in self._clf.model.betas]
        return res

    @betas.setter
    def betas(self, value):
        self._clf.model.betas = [torch.Tensor(b.tondarray()).to(self._device) for b in value]

    @property
    def train_betas(self):
        return self._clf._model.train_betas

    @train_betas.setter
    def train_betas(self, value):
        self._clf.model.train_betas = value

    @property
    def train_prototypes(self):
        return self._clf._model.train_prototypes

    @train_prototypes.setter
    def train_prototypes(self, value):
        self._clf.model.train_prototypes = value

    @property
    def history(self):
        return self._clf._history

    @property
    def _grad_requires_forward(self):       # TODO: Do we need this?! (in CClassifierRejectRBFNet)
        return True


# PARAMETERS
SIGMA = 0.
EPOCHS = 250
BATCH_SIZE = 128


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
    n_hiddens = [250, 250, 50] + [30]  # Combiner init.
    deep_rbf_net = CClassifierDeepRBFNetwork(dnn, layers,
                                             n_hiddens=n_hiddens,
                                             epochs=EPOCHS,
                                             batch_size=BATCH_SIZE,
                                             validation_data=vl_sample,
                                             sigma=SIGMA,  # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                             random_state=random_state)

    # Initialize prototypes with some training samples
    h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    proto = tr_sample[idxs, :]  # HACK: Needed also Y
    deep_rbf_net.prototypes = proto

    # Fit DNR
    deep_rbf_net.verbose = 2  # DEBUG
    deep_rbf_net.fit(tr_sample.X, tr_sample.Y)
    deep_rbf_net.verbose = 0

    # Plot training curves
    fig = plot_train_curves(deep_rbf_net.history, SIGMA)
    fig.savefig("deep_rbf_net_train_sigma_{:.3f}_curves.png".format(SIGMA))

    # Check test performance
    y_pred = deep_rbf_net.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DeepRBFNet Accuracy: {}".format(acc))

    # We can now create a classifier with reject
    clf_rej = CClassifierRejectRBFNet(deep_rbf_net, 0.)

    # Set threshold (FPR: 10%)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Dump to disk
    clf_rej.save('deep_rbf_net_train_sigma_{:.3f}_{}'.format(SIGMA, EPOCHS))