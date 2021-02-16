from scipy.stats import gaussian_kde

from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.array import CArray
from secml.ml.peval.metrics import CMetricAccuracy
from mnist.rbf_net import CClassifierRejectRBFNet

from dataset_loading import load_imagenette, load_imagenet
import matplotlib.pyplot as plt
import numpy as np

CLF = 'rbf_net_classifier:5_100_wd_0e+00_cat_hinge'
LAYER = 'classifier:5'
N_TRAIN, N_TEST = 10000, 100
N_PROTO = 100

random_state = 999
vl = load_imagenette(exclude_val=True)
ts = load_imagenet()

clf = CClassifierRejectRBFNet.load(CLF + '.gz')

# Check test performance
y_pred = clf.predict(ts.X, return_decision_function=False)
perf = CMetricAccuracy().performance_score(ts.Y, y_pred)
print("Model Accuracy: {}".format(perf))

# Select 10K training data and 1K test data (sampling)
tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN,
                            random_state=random_state)
tr_sample = vl[tr_idxs, :]
# HACK: SELECTING VALIDATION DATA (shape=2*N_TEST)
ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST,
                            random_state=random_state)
ts_sample = ts[ts_idxs, :]
vl_sample = ts.deepcopy()

n_hiddens = [N_PROTO]
# Initialize prototypes with some training samples
print("-> Prototypes: Training samples initialization <-")
h = max(n_hiddens)  # HACK: "Nel piu' ci sta il meno..."
proto = CArray.zeros((h, 4096))
n_proto_per_class = h // tr_sample.num_classes
for c in range(tr_sample.num_classes):
    proto[c*n_proto_per_class: (c+1)*n_proto_per_class, :] = \
        clf.clf._features_extractors[LAYER].transform(
            tr_sample.X[tr_sample.Y == c, :][:n_proto_per_class, :])

init_norm = sorted([proto[i, :].norm() for i in range(proto.shape[0])])
trained_norm = sorted([clf.clf.prototypes[0][i, :].norm() for i in
                       range(clf.clf.prototypes[0].shape[0])])

fig, (ax1, ax2) = plt.subplots(2)
colors = ['blue', 'green']
ax1.hist([init_norm, trained_norm], bins=50, color=colors, align="left")
density = gaussian_kde(init_norm)
xs = np.linspace(0, max(init_norm), 200)
ax2.plot(xs, density(xs), color=colors[0])
density = gaussian_kde(trained_norm)
xs = np.linspace(0, max(trained_norm), 200)
ax2.plot(xs, density(xs), colors[1])
fig.legend(["Init", "Trained"])
fig.savefig("prototypes_norm_distribution.pdf")
fig.clear()

nr = CClassifierRejectThreshold.load('nr.gz')

fig = plt.figure(figsize=(30, 10))
for i in range(clf.n_classes):
    nr_scores = nr.decision_function(ts_sample.X)[:, i].ravel().tondarray()
    rbf_scores = clf.decision_function(ts_sample.X)[:, i].ravel().tondarray()
    plt.subplot(2, 6, i + 1)
    plt.scatter(nr_scores, rbf_scores)
    plt.xlabel("NR test scores")
    plt.ylabel("NR-RBF test scores")
    min_score = min(min(nr_scores), min(rbf_scores))
    max_score = max(max(nr_scores), max(rbf_scores))
    plt.xlim([min_score, max_score])
    plt.ylim([min_score, max_score])
fig.tight_layout()
fig.savefig("scores_correlation.pdf")
