import os

from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml import CNormalizerMinMax, CClassifier, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy

from components.torch_nn import MLPytorch
from mnist.deep_rbf_net import CClassifierDeepRBFNetwork
from mnist.rbf_net import CClassifierRBFNetwork

SIGMA = 0.0 # REGULARIZATION KNOB
CLF_TYPE = CClassifierRBFNetwork  # CClassifierDeepRBFNetwork #
CLF_NAME = "CClassifierRBFNetwork" if CLF_TYPE is CClassifierRBFNetwork else "CClassifierDeepRBFNetwork"
N_HIDDENS = [20, 20]
EPOCHS = 10
RUNS = 1
DIR = '{}_net_blobs_sigma_{:.2f}'.format('rbf' if CLF_TYPE is CClassifierRBFNetwork else 'deep_rbf', SIGMA)
os.makedirs(DIR, exist_ok=True)


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


if __name__ == '__main__':
    seed = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 1.2  # Standard deviation of the clusters
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=seed).load()

    tr, ts = CTrainTestSplit(test_size=0.3, random_state=seed).split(ds)

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    ts.X = nmz.transform(ts.X)

    # Load pre-trained classifier
    mlp = MLPytorch(tr.X.shape[1], [128, 128, 128], ds.classes.size,
                    epochs=250,
                    batch_size=32,
                    validation_data=ts,
                    random_state=seed)
    mlp.load_model('mlp_blobs')

    # Create a deep detector for 'dnn'
    n_hiddens = N_HIDDENS
    layers = ['relu2', 'relu3']
    clf = CLF_TYPE(mlp, layers, n_hiddens,
                   epochs=EPOCHS,
                   validation_data=ts,
                   sigma=SIGMA,
                   track_prototypes=False,
                   random_state=seed)

    # Initialize prototypes with some training samples
    h = max(n_hiddens[:-1]) + n_hiddens[-1]  # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr.X.shape[0], shape=(h,), replace=False, random_state=seed)
    proto = tr[idxs, :]
    clf.prototypes = proto

    for run in range(RUNS):

        # Fit clf
        clf.verbose = 1
        clf.fit(tr.X, tr.Y)
        clf.verbose = 0

        # Predict
        y_pred = clf.predict(ts.X)
        acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
        print("[Epoch: {}] Accuracy: {:}".format((run+1)*EPOCHS, acc))

        # if CLF_TYPE is CClassifierRBFNetwork:
        #     pass
        # else:   # Deep case
        #     # Track prototypes (combiner is 2D)
        #     prototypes = []
        #     for p_i in clf._prototypes:
        #         prototypes.append(p_i['combiner'])
        #
        # import numpy as np
        # prototypes = np.array(prototypes)  # shape = (n_tracks, n_hiddens, n_feats)
        # prototypes = [CArray(prototypes[:, i, :]) for i in range(prototypes.shape[1])]

        # Test plot
        from secml.figure import CFigure

        fig = CFigure()

        # # class CClassifierIdentity(CClassifier):
        # #
        # #     def _fit(self, x, y):
        # #         pass
        # #
        # #     def _check_is_fitted(self):
        # #         return True
        # #
        # #     def _forward(self, x):
        # #         return x
        # #
        # #     def _backward(self, w):
        # #         pass
        # #
        # # pre_comb_clf = CClassifierIdentity(preprocess=CNormalizerDNN(clf, "_stack"))
        # # pre_comb_clf._classes = clf._classes
        # # fig.sp.plot_decision_regions(pre_comb_clf, n_grid_points=50, grid_limits=[(-2.5, 2.5), (-2.5, 2.5)])
        # # HACK: Setting 'out_layer' for CClassifierPyTorch
        # clf._out_layer = '_stack'
        # fig.sp.plot_decision_regions(clf, n_grid_points=50, grid_limits=[(-2.5, 2.5), (-2.5, 2.5)])
        # clf._out_layer = None
        #
        # # Register hook
        # activations = None
        # def get_activations(model, input, output):
        #     global activations
        #     activations = output.detach().numpy()
        #
        # clf.model._stack.register_forward_hook(get_activations)
        # # Void Fwd pass
        # clf._batch_size = ts.X.shape[0] # HACK: Setting 'batch_size' to dset shape
        # _ = clf.forward(ts.X)
        # # Retrieve inputs to create new combiner input features dataset
        # fx_dset = CDataset(CArray(activations), ts.Y)
        # fig.sp.plot_ds(fx_dset)
        # # fig.sp.plot_ds(ts)
        #
        # # Plot prototypes
        # for proto in prototypes:
        #     fig.sp.plot_path(proto)

        # DEBUG: PLOT CLF-REJ DECISION REGIONS
        from secml.ml.classifiers.reject import CClassifierRejectThreshold

        clf_rej = CClassifierRejectThreshold(clf, 0.)
        clf_rej.threshold = clf_rej.compute_threshold(0.1, ts)
        fig.sp.plot_decision_regions(clf_rej, n_grid_points=100, grid_limits=[(-2., 2.), (-2., 2)])
        fig.sp.plot_ds(ts)

        fig.title('{} - Sigma {}'.format(CLF_NAME, SIGMA))
        # fig.show()
        fig.savefig(os.path.join(DIR, "{}.png".format(run)))

    # Plot train curves
    fig = plot_train_curves(clf._history, SIGMA)
    fig.savefig(os.path.join(DIR, "train_curves.png"))
    print('done?')
