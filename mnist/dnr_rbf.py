from secml.array import CArray
from secml.core.attr_utils import add_readwrite
from secml.ml import CNormalizerDNN
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy
from torch import nn, optim

from components.rbf_network import RBFNetwork
from components.test_rbf_network import CClassifierPyTorchRBFNetwork
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets
from mnist.rbf_net import rbf_network


class CClassifierDNRBF(CClassifierDNR):

    def __init__(self, dnn, layers, n_hiddens=100, epochs=300, batch_size=32,
                 validation_data=None, threshold=0., random_state=None):

        self._layers = layers

        # Intialize layer_clfs
        self._layer_clfs = {}
        for layer in self._layers:
            self._layer_clfs[layer] = rbf_network(dnn, [layer],
                                                  n_hiddens=n_hiddens,
                                                  epochs=epochs,
                                                  batch_size=batch_size,
                                                  random_state=random_state)
            # TODO: Tracking prototypes? (enabled)
            # search for nested preprocess modules until the inner is reached
            module = self._layer_clfs[layer]
            while module.preprocess is not None:
                module = module.preprocess
            # once the inner preprocess is found, append the dnn to it
            module.preprocess = CNormalizerDNN(net=dnn, out_layer=layer)
            # this allows to access inner classifiers using the
            # respective layer name
            add_readwrite(self, layer, self._layer_clfs[layer])

        # Combiner
        n_feats = len(self._layers) * dnn.n_classes
        model = RBFNetwork(n_feats, n_hiddens, dnn.n_classes)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
        combiner = CClassifierPyTorchRBFNetwork(model,
                                            loss=loss,
                                            optimizer=optimizer,
                                            input_shape=(n_feats, ),
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=validation_data,
                                            track_prototypes=True,  # DEBUG: PROTOTYPES TRACKING ENABLED
                                            random_state=random_state)

        super(CClassifierDNR, self).__init__(combiner, threshold)

    @property
    def verbose(self):
        return super().verbose()

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        # Set to combiner and layer_clfs as well
        for layer in self._layers:
            self._layer_clfs[layer].verbose = value
        self.clf.verbose = value

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

    # Create DNR-RBFNet
    layers = ['features:relu4', 'features:relu3', 'features:relu2']
    dnr_rbf = CClassifierDNRBF(dnn, layers,
                               epochs=1,        # DEBUG
                               threshold=-1000)

    # TODO: SETTING BEST HYPERPARAMS THROUGH XVAL

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr_rbf.verbose = 2  # DEBUG
    dnr_rbf.fit(tr_sample.X, tr_sample.Y)
    dnr_rbf.verbose = 0

    # Check test performance
    y_pred = dnr_rbf.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR-RBF Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr_rbf.threshold = dnr_rbf.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr_rbf.save('dnr_rbf')