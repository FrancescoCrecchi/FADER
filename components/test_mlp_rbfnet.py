from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml import CNormalizerMinMax, CClassifier, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy

from components.torch_nn import MLPytorch
from mnist.deep_rbf_net import CClassifierDeepRBFNetwork


SIGMA = 0.0     # REGULARIZATION KNOB

if __name__ == '__main__':
    seed = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters
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
    n_hiddens = [20, 20, 20]
    layers = ['relu2', 'relu3']
    clf = CClassifierDeepRBFNetwork(mlp, layers,
                                    n_hiddens,
                                    epochs=150,  # DEBUG: RESTORE TO 150 HERE!
                                    # batch_size=128,
                                    # lr=1e-4,
                                    validation_data=ts,
                                    sigma=SIGMA,
                                    track_prototypes=False,
                                    random_state=seed)

    # Initialize prototypes with some training samples
    h = max(n_hiddens[:-1]) + n_hiddens[-1]       # HACK: "Nel piu' ci sta il meno..."
    idxs = CArray.randsample(tr.X.shape[0], shape=(h,), replace=False, random_state=seed)
    proto = tr.X[idxs, :]
    clf.prototypes = proto

    # Fit clf
    clf.verbose = 1
    clf.fit(tr.X, tr.Y)
    clf.verbose = 0

    # Predict
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Accuracy of PyTorch Model: {:}".format(acc))

    # # Track prototypes (combiner is 2D)
    # comb_proto = []
    # for p_i in clf._prototypes:
    #     comb_proto.append(p_i['combiner'])
    #
    # import numpy as np
    # prototypes = np.array(comb_proto)  # shape = (n_tracks, n_hiddens, n_feats)
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
    # # pre_comb_clf = CClassifierIdentity(preprocess=CNormalizerDNN(clf, "_merge"))
    # # pre_comb_clf._classes = clf._classes
    # # fig.sp.plot_decision_regions(pre_comb_clf, n_grid_points=50, grid_limits=[(-2.5, 2.5), (-2.5, 2.5)])
    # # HACK: Setting 'out_layer' for CClassifierPyTorch
    # clf._out_layer = '_merge'
    # fig.sp.plot_decision_regions(clf, n_grid_points=50, grid_limits=[(-2.5, 2.5), (-2.5, 2.5)])
    # clf._out_layer = None
    #
    # # Register hook
    # activations = None
    # def get_activations(model, input, output):
    #     global activations
    #     activations = output.detach().numpy()
    #
    # clf.model._merge.register_forward_hook(get_activations)
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
    fig.sp.plot_decision_regions(clf_rej, n_grid_points=100) #, grid_limits=[(-0.5, 1.5), (-0.5, 1.5)])
    fig.sp.plot_ds(ts)

    fig.title('RBFNetwork Classifier')
    fig.show()
    # fig.savefig('c_classifier_rbf_network.png')

    print('done?')

