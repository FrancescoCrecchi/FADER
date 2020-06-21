import torch
from secml.array import CArray
from secml.ml import CClassifierPyTorch, CClassifier, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy
from torch import nn, optim

from components.rbf_network import RBFNetwork
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


def rbf_network(dnn, layers, n_hiddens=100, epochs=300, batch_size=32, random_state=None):
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # Computing features sizes
    n_feats = [CArray(dnn.get_layer_shape(l)[1:]).prod() for l in layers]
    model = RBFNetwork(n_feats, n_hiddens, dnn.n_classes)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
    return CClassifierPyTorch(model,
                              loss=loss,
                              optimizer=optimizer,
                              input_shape=(sum(n_feats, )),
                              epochs=epochs,
                              batch_size=batch_size,
                              random_state=random_state
                              )


class CClassifierRBFNetwork(CClassifier):

    def __init__(self, dnn, layers, n_hiddens=100, epochs=300, batch_size=32, random_state=None):
        # RBF Network
        self._clf = rbf_network(dnn, layers, n_hiddens, epochs, batch_size, random_state)
        super(CClassifierRBFNetwork, self).__init__()

        # TODO: Parameter Checking

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

    # Create DNR
    layers = ['features:relu4', 'features:relu3', 'features:relu2']
    rbf_net = CClassifierRBFNetwork(dnn, layers, random_state=random_state)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    rbf_net.verbose = 2  # DEBUG

    rbf_net.fit(tr_sample.X, tr_sample.Y)

    # Check test performance
    y_pred = rbf_net.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # # Set threshold (FPR: 10%)
    # dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # # Dump to disk
    # dnr.save('dnr_NEW')
