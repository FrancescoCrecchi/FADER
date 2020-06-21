from secml.array import CArray
from secml.data import CDataset
from secml.ml import CReducer, CNormalizerDNN
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch import nn

from components.torch_nn import MLPytorch


class MyDataset(CDataset):

    def _check_samples_labels(self, x=None, y=None):
        return super()._check_samples_labels(x, y[:, 0])    # HACK TO MAKE IT WORK!

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        x = CArray(value).atleast_2d()
        if self._Y is not None and x.shape[0] != self._Y.shape[0]:
            raise ValueError(
                "number of labels ({:}) must be equal to the number "
                "of samples ({:}).".format(self._Y.size, x.shape[0]))
        self._X = x

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


class CReducerPTSNE(CReducer):

    def __init__(self, n_components=2, n_hiddens=100, epochs=100,
                 batch_size=64, random_state=None, preprocess=None, n_jobs=1):

        super(CReducerPTSNE, self).__init__()

        self.n_components = n_components
        self.n_hiddens = n_hiddens
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._preprocess_ = preprocess
        self._tsne = None
        self._mlp = None

        self._verbose = 0
        self.n_jobs = n_jobs

    def _fit(self, x, y):
        # Fit t-SNE
        self._tsne = TSNE(n_components=self.n_components,
                          method='barnes_hut' if self.n_components < 4 else 'exact',
                          random_state=self.random_state,
                          verbose=self.verbose,
                          n_jobs=self.n_jobs
                          )
        # Produce samples for regression
        if self._preprocess_:
            x_feats = self._preprocess_.forward(x)
        else:
            x_feats = x

        x_embds = CArray(self._tsne.fit_transform(x_feats.tondarray()))
        # DEBUG: Speeding up debugging! TO BE REMOVED!
        # x_embds = CArray.randn(shape=(x_feats.shape[0], self.n_components))
        # x_embds.save('X_tsne', overwrite=True)  # REMOVE THIS!

        # Fit mlp
        nn_in_dim = x_feats.shape[1]    # TODO: CHECK THIS! (High dimensional features?)
        # TODO: Check this below!
        x, x_embds = x.astype('float32'), x_embds.astype('float32')
        # HACK: Use 30% of data for validation performance estimation
        tr_idxs, ts_idxs = train_test_split(CArray.arange(x.shape[0]).tondarray(), test_size=0.3, random_state=self.random_state)
        tr_idxs, ts_idxs = CArray(tr_idxs), CArray(ts_idxs)
        tr_X, tr_Y = x[tr_idxs, :], x_embds[tr_idxs, :]
        vl_X, vl_Y = x[ts_idxs, :], x_embds[ts_idxs, :]
        # Wrap validation data in a 'MyDataset'
        val_data = MyDataset(vl_X, vl_Y)
        self._mlp = MLPytorch(nn_in_dim, self.n_hiddens, self.n_components,
                              loss=nn.MSELoss(), epochs=self.epochs,
                              batch_size=self.batch_size,
                              random_state=self.random_state,
                              preprocess=self._preprocess_,
                              validation_data=val_data
                              )
        self._mlp.verbose = self.verbose
        self._mlp.fit(tr_X, tr_Y)
        self._mlp.verbose = 0

        return self

    def _check_is_fitted(self):
        return self._mlp.is_fitted()

    def _forward(self, x):
        out = self._mlp.forward(x)
        return out

    def _backward(self, w):
        # grad = self._mlp.backward(w)
        grad = self._mlp.get_layer_gradient(x=self._cached_x, w=w)
        return grad

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    from mnist.fit_dnn import get_datasets
    from mnist.cnn_mnist import cnn_mnist_model

    random_state = 999

    tr, vl, ts = get_datasets(random_state)

    # Get dnn
    dnn = cnn_mnist_model()
    dnn.load_model('../mnist/cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Wrap it with `CNormalizerDNN`
    dnn_feats = CNormalizerDNN(dnn, out_layer='features:relu4')

    # Main part
    feat_extr = CReducerPTSNE(
        n_hiddens=[256, 128],
        epochs=1000,
        batch_size=32,
        preprocess=dnn_feats,
        random_state=random_state)

    feat_extr.verbose = 1   # DEBUG
    X_embds = feat_extr.fit_forward(tr_sample.X, tr_sample.Y)

    # Plot
    from secml.data import CDataset
    from secml.figure import CFigure

    fig = CFigure(10, 12)
    fig.sp.plot_ds(CDataset(X_embds, tr_sample.Y))
    fig.sp.legend()
    fig.sp.grid()
    fig.savefig('c_reducer_tnse_mnist.png')

    # # DEBUG: plot original samples
    # X_tsne =  CArray.load('X_tsne')
    # fig = CFigure(10, 12)
    # fig.sp.plot_ds(CDataset(X_tsne, tr_sample.Y))
    # fig.savefig('c_reducer_tnse_mnist_orig.png')

    print("done?")
