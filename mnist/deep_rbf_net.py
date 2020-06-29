import numpy as np
import torch
from torch import nn, optim

from secml.array import CArray
from secml.ml.classifiers.pytorch.c_classifier_pytorch import get_layers
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy

from components.rbf_network import RBFNetwork
from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


class Stack(nn.Module):

    def __init__(self):
        super(Stack, self).__init__()

    def forward(self, iterable, axis):
        x = torch.stack(iterable, axis)
        return x


class DeepRBFNetOnDNN(nn.Module):

    def __init__(self, dnn, layers, input_shape, n_classes, n_hiddens):
        super(DeepRBFNetOnDNN, self).__init__()
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
        self._layer_clfs = nn.ModuleDict()
        # In order to instantiate correctly the RBF module we need to compute the input size
        # we can do this by running a fake sample through the input and looking to the activations sizes
        self._device = next(self.dnn.parameters()).device
        _ = self.dnn(torch.rand(tuple([1] + list(input_shape))).to(self._device))
        i = 0
        for name, layer in get_layers(self.dnn):
            if name in self._layers:
                n_feats = np.prod(self._dnn_activations[layer].shape[1:]).item()
                rbfnet = RBFNetwork(n_feats, self._n_hiddens[i], n_classes)
                # HACK: FIX BETAS
                rbfnet.train_betas = False
                self._layer_clfs[name] = rbfnet
                i += 1
        self._stack = Stack()
        # Set combiner on top
        assert i > 0, "Something wrong in RBFNet layer_clf init!"
        self._combiner = nn.ModuleList()
        for _ in range(n_classes):
            # 1 combiner per class: RBFUnit + LinearUnit -> Class score
            rbfnet = RBFNetwork(len(self._layers), 1, 1)
            # HACK: FIX BETAS
            rbfnet.train_betas = False
            self._combiner.append(rbfnet)

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
                fx.append(self._layer_clfs[name]([activ.view(activ.shape[0], -1)]))
        fx = self._stack(fx, 2)        # fx.shape=(batch_size, n_classes, n_layers)
        # Pass through combiner x class
        out = torch.zeros((x.shape[0], self._n_classes)).to(self._device)
        for c in range(self._n_classes):
            out[:, c] = self._combiner[c]([fx[:, c, :]]).squeeze()

        return out

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     self.dnn = self.dnn.to(*args, **kwargs)
    #     for l in self._layers:
    #         self._layer_clfs[l] = self._layer_clfs[l].to(*args, **kwargs)
    #     self._combiner = self._combiner.to(*args, **kwargs)
    #     return self


class CClassifierDeepRBFNetwork(CClassifierPyTorchRBFNetwork):

    def __init__(self, dnn, layers, n_hiddens=100,
                 epochs=300, batch_size=32,
                 lr=1e-3,
                 validation_data=None,
                 sigma=0.,
                 track_prototypes=False,
                 random_state=None):

        # Param checking
        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens] * (len(layers))  # Taking care of the combiner..
        self._n_hiddens = n_hiddens

        # DeepRBFNetOnDNN (TODO: pass other params)
        model = DeepRBFNetOnDNN(dnn.model, layers, dnn.input_shape, dnn.n_classes, self._n_hiddens)
        # Loss & Optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  # --> TODO: Expose optimizer params <--
        super(CClassifierDeepRBFNetwork, self).__init__(model,
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
            proto['layer_clfs'][l] = self.model._layer_clfs[l].prototypes[0].clone().detach().numpy()
        # Combiner prototypes
        proto['combiner'] = self.model._combiner.prototypes[0].clone().detach().numpy()
        return proto

    @prototypes.setter
    def prototypes(self, dset):
        '''
        Set 'DeepRBFNetOnDNN' layer_clfs and combiner prototypes (same Xs for all)
        :param value: Input prototypes vectors
        '''
        # Unpack and reshape
        x, y = dset.X.tondarray().reshape(-1, *self.input_shape), dset.Y.tondarray()

        # Move model to 'device'
        self.model.to(self._device)

        # Convert to torch.Tensor
        x_torch = torch.Tensor(x).float().to(self._device)

        # Void run to compute hooks
        _ = self.model.dnn.forward(x_torch)
        # Use computed features to setup prototypes
        i = 0
        for name, layer in get_layers(self.model.dnn):
            if name in self._layers:
                activ = self.model._dnn_activations[layer][:self._n_hiddens[i]]
                self.model._layer_clfs[name].prototypes = [activ.view(activ.shape[0], -1)]
                i += 1

        # Select one sample per class to init. combiner prototypes
        comb_x = torch.zeros((self._n_classes, *x_torch.shape[1:]))
        for c in range(self._n_classes):
            # Selecting the fist one, for simplicity
            comb_x[c, :] = x_torch[y == c][0]

        # Run dnn on them
        _ = self.model.dnn.forward(comb_x.to(self._device))
        # Pack activations
        fx = []
        i = 0
        for name, layer in get_layers(self.model.dnn):
            if name in self._layers:
                activ = self.model._dnn_activations[layer][:self._n_hiddens[i]]
                out = self.model._layer_clfs[name]([activ.view(activ.shape[0], -1)])
                fx.append(out)
                i += 1
        fx = torch.stack(fx, 2)
        for c in range(self._n_classes):
            self.model._combiner[c].prototypes = [fx[c, c, :][None, :]]

    # TODO: Expose Betas


SIGMA = 0.0         # HACK: REGULARIZATION KNOB

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
    n_hiddens = [250, 250, 50]
    rbf_net = CClassifierDeepRBFNetwork(dnn, layers,
                                        n_hiddens=n_hiddens,
                                        epochs=40,
                                        batch_size=32,
                                        validation_data=vl_sample,
                                        sigma=SIGMA,  # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                        random_state=random_state)

    # Initialize prototypes with some training samples
    h = max(n_hiddens[:-1]) + n_hiddens[-1]       # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    proto = tr_sample[idxs, :]
    rbf_net.prototypes = proto

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
    # clf_rej.save('deep_rbf_net_sigma_{}'.format(SIGMA))