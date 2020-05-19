# from setGPU import setGPU
# setGPU(3)

import os

from secml.array import CArray
from secml.ml import CReducer, CNormalizerDNN
from sklearn.manifold import TSNE
from torch import nn

from mnist import mnist
from ptSNE import scatter_plot
from torch_nn import MLPytorch


class CReducerPTSNE(CReducer):

    def __init__(self,
                 n_components=2,
                 n_hiddens=100,
                 epochs=100,
                 batch_size=64,
                 verbose=0,
                 random_state=None,
                 preprocess=None):

        self.n_components = n_components
        self.n_hiddens = n_hiddens
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        self._preprocess_ = preprocess
        self._tsne = None
        self._mlp = None

        super().__init__()  # preprocess)

    def _fit(self, x, y):
        # Check preprocess has type `CClassifierDNN`
        if self._preprocess_ and self._preprocess_.__class__ is CNormalizerDNN:
            # Computing 'out_layer' feature dims
            lidx = [l[0] for l in self._preprocess_.net.layers].index(self._preprocess_.out_layer)
            out_feats = self._preprocess_.net.layers[lidx][1].out_features
        else:
            out_feats = x.shape[1]

        # Fit t-SNE
        self._tsne = TSNE(n_components=self.n_components,
                          method='barnes_hut' if self.n_components < 4 else 'exact',
                          random_state=self.random_state,
                          verbose=self.verbose
                          )
        # Produce samples for regression
        if self._preprocess_:
            x_feats = self._preprocess_.forward(x)
        else:
            x_feats = x
        x_embds = CArray(self._tsne.fit_transform(x_feats.tondarray()))

        # Fit mlp
        self._mlp = MLPytorch(out_feats, self.n_hiddens, self.n_components,
                              loss=nn.MSELoss(), epochs=self.epochs,
                              batch_size=self.batch_size,
                              random_state=self.random_state,
                              preprocess=self._preprocess_)
        self._mlp.verbose = self.verbose
        self._mlp.fit(x.astype('float32'), x_embds.astype('float32'))

        return self

    def _check_is_fitted(self):
        return self._mlp.is_fitted()

    def _forward(self, x):
        out = self._mlp.forward(x)
        return out

    def _backward(self, w):
        # TODO: CHECK THIS!
        grad = self._mlp.backward(w)
        return grad
    

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
    feat_extr = CReducerPTSNE(preprocess=dnn_feats,
                              random_state=random_state,
                              verbose=1)
    feat_extr.fit(sample.X)
    X_embds = feat_extr.transform(sample.X)

    # Plot
    from secml.figure import CFigure

    fig = CFigure(10, 12)
    scatter_plot(fig.sp, X_embds, sample.Y)
    fig.sp.legend()
    fig.sp.grid()
    fig.savefig('c_reducer_tnse_mnist.png')

    print("done?")
