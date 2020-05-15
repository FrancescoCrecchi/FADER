import os

from sklearn.manifold import TSNE
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerDNN
import torch
from torch import nn, optim

from mnist import mnist, Flatten



# class MyNormalizerDNN(CNormalizerDNN):
#
#     @property
#     def classes(self):
#         return self.net.classes


class CRegressorPytorch(CClassifierPyTorch):

    def fit(self, dataset, n_jobs=1):
        """Same as `CClassifier.fit` but with `MyDataset` instead of `CDataset`

                If a preprocess has been specified,
                input is normalized before training.

                For multiclass case see `.CClassifierMulticlass`.

                Parameters
                ----------
                dataset : MyDataset
                    Training set. Must be a :class:`.MyDataset` instance with
                    patterns data and corresponding labels.
                n_jobs : int
                    Number of parallel workers to use for training the classifier.
                    Default 1. Cannot be higher than processor's number of cores.

                Returns
                -------
                trained_cls : CClassifier
                    Instance of the classifier trained using input dataset.

                """
        if not isinstance(dataset, CDataset):
            raise TypeError(
                "training set should be provided as a CDataset object.")

        data_x, data_y = dataset.X, dataset.Y
        # Transform data if a preprocess is defined
        if self.preprocess is not None:
            data_x = self.preprocess.fit_transform(data_x)

        # Storing dataset classes
        # TODO: CHECK POSSIBLE NON-FLAT SHAPES!
        self._classes = data_y.shape[1]
        self._n_features = data_x.shape[1]

        # Data is ready: fit the classifier
        try:  # Try to use parallelization
            self._fit(data_x, data_y, n_jobs=n_jobs)
        except TypeError:  # Parallelization is probably not supported
            self._fit(data_x, data_y)

        return self


class MyDataset(CDataset):

    def _check_samples_labels(self, x=None, y=None):
        return super()._check_samples_labels(x, y[:, 0])  # HACK TO MAKE IT WORK!

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        """Set Dataset Labels.

        Parameters
        ----------
        value : `array_like` or CArray
            Array containing labels.
        """
        y = CArray(value).todense()  # .ravel()
        if self._X is not None:  # Checking number of samples/labels equality
            self._check_samples_labels(y=y)
        self._Y = y


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.flat = Flatten()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, D_out)

    def forward(self, x):
        h1 = torch.relu(self.fc1(self.flat(x)))
        h2 = torch.relu(self.fc2(h1))
        y = self.fc3(h2)
        return y


def ptSNE(dset, d=2,
          verbose=0,
          hidden_size=100,
          epochs=500,
          batch_size=64,
          random_state=None,
          preprocess=None):
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # Unpack data
    X, y = dset.X, dset.Y
    # Computing 'out_layer' feature dims
    lidx = [l[0] for l in preprocess.net.layers].index(preprocess.out_layer)
    out_feats = preprocess.net.layers[lidx][1].out_features
    # Setting DNN params
    D_in = out_feats
    H = hidden_size
    D_out = d
    # Create DNN
    dnn = Net(D_in, H, D_out)
    # Compute tSNE mappings
    X_embds = CArray(TSNE(n_components=d,
                          random_state=random_state,
                          verbose=verbose,
                          method='barnes_hut' if d < 4 else 'exact'
                          ).fit_transform(X.tondarray()))
    # X_embds = CArray.randn((X.shape[0], d)) # TODO: REMOVE THIS!
    # Wrap `dnn` in a `CRegressorPytorch` and fit
    dnn = CRegressorPytorch(dnn,
                            loss=nn.MSELoss(),
                            optimizer=optim.SGD(dnn.parameters(), lr=1e-3),
                            epochs=epochs,
                            batch_size=batch_size,
                            input_shape=(out_feats,),
                            preprocess=preprocess)
    tr = MyDataset(X.astype('float32'), X_embds.astype('float32'))
    dnn.verbose = verbose
    dnn.fit(tr)
    dnn.verbose = 0
    # Wrap it in a `CNormalizerDNN` as it is actually a features extractor
    feat_extr = CNormalizerDNN(dnn, out_layer='fc3')

    return feat_extr, X_embds


# def ptSNE(dset, d=2, random_state=None, verbose=0, epochs=500, batch_size=64, preprocess=None):
#     use_cuda = torch.cuda.is_available()
#     if random_state is not None:
#         torch.manual_seed(random_state)
#     if use_cuda:
#         torch.backends.cudnn.deterministic = True
#     # Unpack data
#     X, y = dset.X, dset.Y
#     # Computing 'out_layer' feature dims
#     lidx = [l[0] for l in preprocess.net.layers].index(preprocess.out_layer)
#     out_feats = preprocess.net.layers[lidx][1].out_features
#     # Setting DNN params
#     D_in = out_feats
#     H = 1000
#     D_out = d
#     # Create DNN
#     dnn = TwoLayerNet(D_in, H, D_out)
#     # Compute tSNE mappings
#     # X_embds = CArray(TSNE(n_components=d, random_state=random_state, verbose=verbose).fit_transform(X.tondarray()))
#     X_embds = CArray.randn((X.shape[0], d)) # TODO: REMOVE THIS!
#     # Wrap `dnn` in a `CRegressorPytorch` and fit
#     dnn = CRegressorPytorch(dnn,
#                             loss=nn.MSELoss(),
#                             optimizer=optim.SGD(dnn.parameters(), lr=1e-3),
#                             epochs=epochs,
#                             batch_size=batch_size,
#                             input_shape=(out_feats,),
#                             preprocess=None)
#     # Split forward pass
#     X = preprocess.transform(X)
#     tr = MyDataset(X.astype('float32'), X_embds.astype('float32'))
#     dnn.verbose = verbose
#     dnn.fit(tr)
#     # Wrap it in a `CNormalizerDNN` as it is actually a features extractor
#     feat_extr = CNormalizerDNN(dnn, out_layer='fc2')
#
#     return feat_extr

def scatter_plot(sp, X, y):
    assert len(X.shape) == 2, "X must be 2d!"
    import numpy as np
    import matplotlib.cm as cm

    colors = np.array(cm.tab10.colors)
    for i in range(10):
        cl_idxs = CArray(np.where(y.tondarray() == i)[0])
        sp.scatter(X[cl_idxs, 0], X[cl_idxs, 1], label='{}'.format(i), alpha=.7, c=colors[i][None, :])


N_TRAIN = 30000
if __name__ == '__main__':
    from setGPU import setGPU
    setGPU(-1)

    random_state = 999

    # Prepare data
    from secml.data.loader import CDataLoaderMNIST

    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    # Normalize the data
    tr.X /= 255

    # Get dnn
    dnn = mnist()
    if not os.path.exists("mnist.pkl"):
        dnn.verbose = 1
        dnn.fit(tr[:N_TRAIN, :])
        dnn.save_model("mnist.pkl")
    else:
        dnn.load_model("mnist.pkl")

    # Wrap it with `CNormalizerDNN`
    dnn_feats = CNormalizerDNN(dnn, out_layer='fc3')

    # Main part
    sample = tr[N_TRAIN:N_TRAIN + 3000, :]
    feat_extr, X_tsne = ptSNE(sample,
                              d=2,
                              hidden_size=64,
                              epochs=1000,
                              batch_size=128,
                              preprocess=dnn_feats,
                              random_state=random_state,
                              verbose=1)
    X_embds = feat_extr.transform(sample.X)

    # Plot
    from secml.figure import CFigure

    fig = CFigure(10, 24)
    # Orig. plots
    sp1 = fig.subplot(1, 2, 1)
    scatter_plot(sp1, X_tsne, sample.Y)
    sp1.legend()
    sp1.grid()
    # NN plots
    sp2 = fig.subplot(1, 2, 2)
    scatter_plot(sp2, X_embds, sample.Y)
    sp2.legend()
    sp2.grid()
    fig.savefig('ptSNE_mnist.png')

    # Test gradient
    x = sample.X[:10, :]
    w = feat_extr.forward(x)
    grad = feat_extr.gradient(x, w=w)
    print(grad.shape)

    # # ========= SPLITTING =========
    # # feat_extr.preprocess = None
    # # Forward
    # e = dnn_feats.forward(x)
    # y = feat_extr.forward(e)
    # # Backward
    # de = feat_extr.gradient(e, y)
    # dx = dnn_feats.gradient(x, de)

    # TODO: CHECK GRADIENT! (SOMETHING WRONG HERE!)
    # # Numerical gradient check
    # from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases
    #
    # CClassifierTestCases.setUpClass()
    # CClassifierTestCases()._test_gradient_numerical(feat_extr, sample.X[0, :])

    print("done?")
