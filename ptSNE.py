# from setGPU import setGPU
# setGPU()

import os
import torch
from torch import nn
from sklearn.manifold import TSNE

from secml.array import CArray
from secml.ml.features import CNormalizerDNN

from mnist import mnist
from torch_nn import MLPytorch


# class CRegressorPytorch(CClassifierPyTorch):
#
#     def _check_input(self, x, y=None):
#         return x, y
#
#     def _fit(self, x, y):
#
#         # Storing dataset classes
#         # TODO: CHECK POSSIBLE NON-FLAT SHAPES!
#         self._n_features = x.shape[1]
#         self._classes = CArray.arange(y.shape[1])
#
#         return super()._fit(x, y)


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
    # Compute tSNE mappings
    X_embds = CArray(TSNE(n_components=d,
                          random_state=random_state,
                          verbose=verbose,
                          method='barnes_hut' if d < 4 else 'exact'
                          ).fit_transform(preprocess.forward(X).tondarray()))
    # X_embds = CArray.randn((X.shape[0], d))  # TODO: REMOVE THIS!
    # Wrap `dnn` in a `CRegressorPytorch` and fit
    dnn = MLPytorch(D_in, H, D_out,
                    loss=nn.MSELoss(),
                    epochs=epochs,
                    batch_size=batch_size,
                    preprocess=preprocess)
    dnn.verbose = verbose
    # TODO: perform xval here!
    dnn.fit(X.astype('float32'), X_embds.astype('float32'))
    dnn.verbose = 0
    # Wrap it in a `CNormalizerDNN` as it is actually a features extractor
    feat_extr = CNormalizerDNN(dnn, out_layer='out')

    return feat_extr, X_embds


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
                              hidden_size=[64, 64],
                              epochs=500,
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
    # Numerical gradient check
    from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases

    CClassifierTestCases.setUpClass()
    CClassifierTestCases()._test_gradient_numerical(feat_extr.net, sample.X[0, :])

    print("done?")
