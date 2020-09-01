# Basic separability test for DNN WB attack samples

from secml.adv.attacks import CAttackEvasionPGD
from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml import CNormalizerDNN
from sklearn.manifold import TSNE

from cifar10.fit_dnn import get_datasets

DSET = 'cifar10'
N_SAMPLES = 1000
MARGIN = 300
EPS = 0.5

# Load 1000 samples
random_state = 999
_, vl, _ = get_datasets(random_state)

# Select a sample from dset
sample = vl[:N_SAMPLES+MARGIN, :]  # 300 is a margin for non-evading samples

# Load attack
pgd_attack = CAttackEvasionPGD.load('dnn_attack.gz')
# Setting max distance
pgd_attack.dmax = EPS
# Run evasion
pgd_attack.verbose = 2 # DEBUG
eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(sample.X, sample.Y)

# Select effective evading samples
evading_samples = eva_adv_ds[eva_y_pred != sample.Y, :]
N = min(evading_samples.X.shape[0], N_SAMPLES)
evading_samples = evading_samples[:N, :]

X_nat, y_nat = sample[:N_SAMPLES, :].X, sample[:N_SAMPLES, :].Y + 1
X_adv, y_adv = evading_samples.X, -(evading_samples.Y + 1)    # Negating to differentiate natural from adv. samples

# Pass through features extractor
feat_extr = CNormalizerDNN(pgd_attack.classifier, out_layer='features:29')
X_embds = feat_extr.forward(CArray.concatenate(X_nat, X_adv, axis=0))

# TSNE part
X_2d = TSNE().fit_transform(X_embds.tondarray())
y_2d = CArray.concatenate(y_nat, y_adv)

# Plot separability with TSNE
foo = CDataset(X_2d, y_2d)

fig = CFigure(height=8, width=10)
REF_CLASS = 2
foo_ref = foo[(foo.Y == REF_CLASS+1).logical_or(foo.Y == -(REF_CLASS+1)), :]
fig.sp.plot_ds(foo_ref)
# fig.sp.plot_ds(foo)
fig.savefig('tsne_adv.png')