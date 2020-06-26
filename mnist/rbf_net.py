from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from secml.array import CArray
from secml.ml.classifiers.pytorch.c_classifier_pytorch import get_layers, CClassifierPyTorch, use_cuda
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy

from components.rbf_network import RBFNetwork
from components.test_rbf_network import CClassifierPyTorchRBFNetwork

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


class RBFNetOnDNN(nn.Module):

    def __init__(self, dnn, layers, input_shape, n_classes, n_hiddens):
        super(RBFNetOnDNN, self).__init__()
        self._layers = layers
        self._n_hiddens = n_hiddens
        self._n_classes = n_classes
        # DNN
        self.dnn = dnn
        # Freeze DNN layers (assuming pretrained)
        for param in self.dnn.parameters():
            param.requires_grad = False
        self._register_hooks()
        # RBFNet
        # Setting layer_clfs
        self._layer_clfs = {}
        # In order to instantiate correctly the RBF module we need to compute the input size
        # we can do this by running a fake sample through the input and looking to the activations sizes
        _ = self.dnn(torch.rand(tuple([1] + list(input_shape))).to('cuda'))
        i = 0
        for name, layer in get_layers(self.dnn):
            if name in self._layers:
                n_feats = np.prod(self._dnn_activations[layer].shape[1:]).item()
                self._layer_clfs[name] = RBFNetwork(n_feats, self._n_hiddens[i], n_classes)
                i += 1
        # Set combiner ontop
        assert i > 0, "Something wrong in RBFNet layer_clf init!"
        # n_feats = len(self._layers) * n_classes
        # self._combiner = nn.Sequential(OrderedDict([
        #     ('batch_norm', nn.BatchNorm1d(n_feats)),
        #     ('combiner', RBFNetwork(n_feats, self._n_hiddens[i], n_classes))
        # ]))
        n_feats = sum(self._n_hiddens[:-1])
        self._combiner = nn.Linear(n_feats, self._n_classes)

    def _register_hooks(self):
        # ========= Setting hooks =========
        # FWD >>
        self._handlers = {}
        self._dnn_activations = {}

        for name, layer in get_layers(self.dnn):
            if name in self._layers:
                self._handlers[name] = layer.register_forward_hook(self.get_activation)
        # ===============================

    def get_activation(self, module_name, input, output):
        self._dnn_activations[module_name] = output

    def forward(self, x):
        _ = self.dnn(x)  # This sets layer activations
        fx = []
        for name, layer in get_layers(self.dnn):
            if name in self._layers:
                activ = self._dnn_activations[layer]
                fx.append(self._layer_clfs[name](activ.view(activ.shape[0], -1)))
        fx = torch.cat(fx, 1)

        out = self._combiner(fx)
        return out

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.dnn = self.dnn.to(*args, **kwargs)
        for l in self._layers:
            self._layer_clfs[l] = self._layer_clfs[l].to(*args, **kwargs)
        self._combiner = self._combiner.to(*args, **kwargs)
        return self


class CClassifierRBFNetwork(CClassifierPyTorchRBFNetwork):

    def __init__(self, dnn, layers, n_hiddens=100,
                 epochs=300, batch_size=32,
                 validation_data=None,
                 sigma=0.,
                 track_prototypes=False,
                 random_state=None):

        # Param checking
        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens] * (len(layers) + 1)  # Taking care of the combiner..
        self._n_hiddens = n_hiddens

        # RBFNetOnDNN (TODO: pass other params)
        model = RBFNetOnDNN(dnn.model, layers, dnn.input_shape, dnn.n_classes, self._n_hiddens)
        # Loss & Optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())  # --> TODO: Expose optimizer params <--
        super(CClassifierRBFNetwork, self).__init__(model,
                                                    loss=loss,
                                                    optimizer=optimizer,
                                                    input_shape=dnn.input_shape,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    validation_data=validation_data,
                                                    track_prototypes=track_prototypes,
                                                    sigma=sigma,
                                                    random_state=random_state)

        # Internals
        self._layers = layers
        self._n_classes = dnn.classes.size

    @property
    def prototypes(self):
        proto = {'layer_clfs': {}}
        # Layer clfs prototypes
        for l in self._layers:
            proto['layer_clfs'][l] = self.model._layer_clfs[l].prototypes
        # # Combiner prototypes
        # proto['combiner'] = self.model._combiner.combiner.prototypes
        return proto

    @prototypes.setter
    def prototypes(self, x):
        '''
        Set 'RBFNetOnDNN' layer_clfs and combiner prototypes (same Xs for all)
        :param value: Input prototypes vectors
        '''
        # Move model to 'device'
        self.model.to(self._device)

        # Convert to torch.Tensor
        x = torch.Tensor(x.tondarray().reshape(-1, *self.input_shape)).float().to(self._device)

        # Hold-out 'n_hiddens[-1]' samples at random to avoid combiner overfitting
        idxs = torch.Tensor(CArray.randsample(x.shape[0], self._n_hiddens[-1], random_state=random_state).tondarray())
        b_mask = torch.zeros(x.shape[0])
        b_mask[idxs.long()] = 1.0
        b_mask = b_mask.bool()
        x_comb = x[b_mask, :]
        x = x[~b_mask, :]

        # Void run to compute hooks
        _ = self.model.dnn.forward(x)
        # Use computed features to setup prototypes
        i = 0
        for name, layer in get_layers(self.model.dnn):
            if name in self._layers:
                activ = self.model._dnn_activations[layer][:self._n_hiddens[i]]
                self.model._layer_clfs[name].prototypes = [activ.view(activ.shape[0], -1)]
                i += 1

        # # Run dnn on them
        # _ = self.model.dnn.forward(x_comb)
        # # Pack activations
        # fx = []
        # i = 0
        # for name, layer in get_layers(self.model.dnn):
        #     if name in self._layers:
        #         activ = self.model._dnn_activations[layer][:self._n_hiddens[i]]
        #         out = self.model._layer_clfs[name](activ.view(activ.shape[0], -1))
        #         fx.append(out)
        #         i += 1
        # fx = torch.cat(fx, 1)
        # self.model._combiner.combiner.prototypes = [fx]

    # TODO: Expose Betas


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
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=2 * N_TEST, random_state=random_state)
    vl_sample = ts[ts_idxs[:N_TEST], :]
    ts_sample = ts[ts_idxs[N_TEST:], :]

    # Create DNR
    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    n_hiddens = [250, 250, 50, 10]
    rbf_net = CClassifierRBFNetwork(dnn, layers,
                                    n_hiddens=n_hiddens,
                                    epochs=100,
                                    batch_size=32,
                                    validation_data=vl_sample,
                                    # sigma=1.0,  # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                    random_state=random_state)

    # # Initialize prototypes with some training samples
    # h = max(n_hiddens[:-1]) + n_hiddens[-1]       # HACK: "Nel piu' ci sta il meno..."
    # idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    # proto = tr_sample.X[idxs, :]
    # rbf_net.prototypes = proto

    # Fit DNR
    rbf_net.verbose = 2  # DEBUG
    rbf_net.fit(tr_sample.X, tr_sample.Y)
    rbf_net.verbose = 0

    # Check test performance
    y_pred = rbf_net.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("RBFNet Accuracy: {}".format(acc))

    # # We can now create a classifier with reject
    # clf_rej = CClassifierRejectThreshold(rbf_net, 0.)
    #
    # # Set threshold (FPR: 10%)
    # clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)
    #
    # # Dump to disk
    # clf_rej.save('rbf_net')
