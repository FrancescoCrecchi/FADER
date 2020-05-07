from sklearn.manifold import TSNE
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerDNN
import torch
from torch import nn, optim


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

        # Storing dataset classes
        self._classes = dataset.classes
        self._n_features = dataset.num_features

        data_x = dataset.X
        # Transform data if a preprocess is defined
        if self.preprocess is not None:
            data_x = self.preprocess.fit_transform(dataset.X)

        # Data is ready: fit the classifier
        try:  # Try to use parallelization
            self._fit(MyDataset(data_x, dataset.Y), n_jobs=n_jobs)
        except TypeError:  # Parallelization is probably not supported
            self._fit(MyDataset(data_x, dataset.Y))

        return self


class MyDataset(CDataset):

    def _check_samples_labels(self, x=None, y=None):
        return super()._check_samples_labels(x, y[:, 0])    # HACK TO MAKE IT WORK!

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
        y = CArray(value).todense()#.ravel()
        if self._X is not None:  # Checking number of samples/labels equality
            self._check_samples_labels(y=y)
        self._Y = y


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        y = self.fc2(h)
        return y


def ptSNE(X, d=2, random_state=None, verbose=0):
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    # Input
    D_in = X.shape[1]
    H = 1000
    D_out = d
    # Create DNN
    dnn = TwoLayerNet(D_in, H, D_out)
    # Compute tSNE mappings
    X_embds = CArray(TSNE(n_components=d, random_state=random_state, verbose=verbose).fit_transform(X.tondarray()))
    # Wrap `dnn` in a `CRegressorPytorch` and fit
    dnn = CRegressorPytorch(dnn,
                            loss=nn.MSELoss(),
                            optimizer=optim.SGD(dnn.parameters(), lr=1e-3),
                            epochs=500,
                            batch_size=64,
                            input_shape=(D_in,))
    tr = MyDataset(X.astype('float32'), X_embds.astype('float32'))
    dnn.verbose = 1
    dnn.fit(tr)
    # Wrap it in a `CNormalizerDNN` as it is actually a features extractor
    feat_extr = CNormalizerDNN(dnn, out_layer='fc2')

    return feat_extr


if __name__ == '__main__':
    import setGPU
    random_state = 999

    # Prepare data
    from secml.data.loader import CDataLoaderMNIST
    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    # Normalize the data
    tr.X /= 255

    # Main part
    X_sample, y_sample = tr.X[:3000, :], tr.Y[:3000]
    feat_extr = ptSNE(X_sample, d=2, random_state=random_state, verbose=1)
    X_embds = feat_extr.transform(X_sample)

    from secml.figure import CFigure
    import numpy as np
    import matplotlib.cm as cm

    fig = CFigure(10, 12)
    colors = np.array(cm.tab10.colors)
    for i in range(10):
        cl_idxs = CArray(np.where(y_sample.tondarray() == i)[0])
        fig.sp.scatter(X_embds[cl_idxs, 0], X_embds[cl_idxs, 1], label='{}'.format(i), alpha=.7, c=colors[i][None, :])

    fig.sp.legend()
    fig.sp.grid()
    fig.savefig('ptSNE_mnist.png')

    print("done?")

