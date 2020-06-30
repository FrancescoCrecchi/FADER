import numpy as np
import torch
from torch import nn, optim

from secml.array import CArray
from secml.ml.classifiers.pytorch.c_classifier_pytorch import get_layers
from secml.ml import CClassifierPyTorch, CClassifier, CNormalizerDNN
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetricAccuracy
from secml.figure import CFigure

from components.c_classifier_pytorch_rbf_network import CClassifierPyTorchRBFNetwork
from components.rbf_network import RBFNetwork

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


# def rbf_network(dnn, layers, n_hiddens=100, epochs=300, batch_size=32, validation_data=None, sigma=1.0,
#                 random_state=None):
#     # Use CUDA
#     use_cuda = torch.cuda.is_available()
#     if random_state is not None:
#         torch.manual_seed(random_state)
#     if use_cuda:
#         torch.backends.cudnn.deterministic = True
#
#     # RBFNetOnDNN (TODO: pass other params)
#     model = RBFNetOnDNN(dnn.model, layers, dnn.input_shape, dnn.n_classes, n_hiddens)
#     # Loss & Optimizer
#     loss = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.rbfnet.parameters())  # --> TODO: Expose optimizer params <--
#     # HACK: TRACKING PROTOTYPES
#     return CClassifierRBFNetwork(model,
#                                  loss=loss,
#                                  optimizer=optimizer,
#                                  input_shape=dnn.input_shape,
#                                  epochs=epochs,
#                                  batch_size=batch_size,
#                                  validation_data=validation_data,
#                                  # track_prototypes=True,  # DEBUG: PROTOTYPES TRACKING ENABLED
#                                  # sigma=sigma,
#                                  random_state=random_state)


def plot_train_curves(history, sigma):
    fig = CFigure()
    fig.sp.plot(history['tr_loss'], label='TR', marker="o")
    fig.sp.plot(history['vl_loss'], label='VL', marker="o")
    fig.sp.plot(history['xentr_loss'], label='xentr', marker="o")
    fig.sp.plot(history['reg_loss'], label='reg', marker="o")
    fig.sp.plot(history['weight_decay'], label='decay', marker="o")
    fig.sp.title("Training Curves - Sigma: {}".format(sigma))
    fig.sp.legend()
    fig.sp.grid()
    return fig


class RBFNetOnDNN(nn.Module):

    def __init__(self, dnn, layers, input_shape, n_classes, n_hiddens=100):
        super(RBFNetOnDNN, self).__init__()
        self._layers = layers
        # DNN
        self.dnn = dnn
        # Freeze DNN layers (assuming pretrained)
        for param in self.dnn.parameters():
            param.requires_grad = False
        self._register_hooks()
        # RBFNet
        # In order to instantiate correctly the RBF module we need to compute the input size
        # we can do this by running a fake sample through the input and looking to the activations sizes
        dnn_device = next(self.dnn.parameters()).device
        _ = self.dnn(torch.rand(tuple([1] + list(input_shape))).to(dnn_device))
        n_feats = []
        for name, layer in get_layers(self.dnn):
            if name in self._layers:
                n_feats.append(np.prod(self._dnn_activations[layer].shape[1:]))
        # n_feats =s [np.prod(list(self._dnn_activations[l].shape[1:])) for l in self._layers]
        self.rbfnet = RBFNetwork(n_feats, n_hiddens, n_classes)
        # HACK: FIX BETAS
        self.rbfnet.train_betas = False

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
                fx.append(activ.view(activ.shape[0], -1))
        # Passing a list of activations
        out = self.rbfnet(fx)
        return out


class CClassifierRBFNetwork(CClassifierPyTorchRBFNetwork):

    def __init__(self, dnn, layers, n_hiddens=100,
                 epochs=300, batch_size=32,
                 validation_data=None,
                 sigma=0.0,     # DEFAULT: No regularization!
                 track_prototypes=False,
                 random_state=None):
        # Param checking
        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens] * len(layers)
        self._n_hiddens = n_hiddens

        # RBFNetOnDNN (TODO: pass other params)
        model = RBFNetOnDNN(dnn.model, layers, dnn.input_shape, dnn.n_classes, self._n_hiddens)
        # Loss & Optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters()) #, weight_decay=1e-3)  # --> TODO: Expose optimizer params <--
        # optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-2)
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

    @property
    def prototypes(self):
        res = [CArray(proto.clone().detach().cpu().numpy()) for proto in self.model.rbfnet.prototypes]
        return res

    @prototypes.setter
    def prototypes(self, dset):

        # Unpack and reshape
        x = dset.X.tondarray().reshape(-1, *self.input_shape)

        # Move model to 'device'
        self.model.to(self._device)
        # Convert to torch.Tensor
        x_torch = torch.Tensor(x).float().to(self._device)
        # Void run to compute hooks
        _ = self.model.dnn.forward(x_torch)
        # Use computed features to setup prototypes
        i = 0
        fx = []
        for name, layer in get_layers(self.model.dnn):
            if name in self._layers:
                activ = self.model._dnn_activations[layer][:self._n_hiddens[i]]
                fx.append(activ.view(activ.shape[0], -1))
                i += 1
        self.model.rbfnet.prototypes = fx

    # TODO: Expose Betas


SIGMA = 0.0         # HACK: REGULARIZATION KNOB
EPOCHS = 30

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
    rbf_net = CClassifierRBFNetwork(dnn, layers,
                                    n_hiddens=n_hiddens,
                                    epochs=EPOCHS,
                                    batch_size=32,
                                    validation_data=vl_sample,
                                    sigma=SIGMA,  # TODO: HOW TO SET THIS?! (REGULARIZATION KNOB)
                                    random_state=random_state)

    # Initialize prototypes with some training samples
    h = max(n_hiddens)       # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr_sample.X.shape[0], shape=(h,), replace=False, random_state=random_state)
    proto = tr_sample[idxs, :]
    rbf_net.prototypes = proto

    # Fit DNR
    rbf_net.verbose = 1  # DEBUG
    rbf_net.fit(tr_sample.X, tr_sample.Y)
    rbf_net.verbose = 0

    # Plot training curves
    fig = plot_train_curves(rbf_net._history, SIGMA)
    fig.savefig("rbf_net_train_curves.png")

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
    # clf_rej.save('rbf_net_sigma_{}_{}'.format(SIGMA, EPOCHS))