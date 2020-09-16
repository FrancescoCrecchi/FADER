import os

from secml.ml.classifiers.reject import CClassifierDNR, CClassifierRejectThreshold

# PARAMETERS
DSET = 'cifar10'
CLF = 'nr'


def count_svm_prototypes(svm):
    sv = abs(svm._alpha).sum(axis=0) > 0
    return sv.sum()

fname = os.path.join(DSET, CLF+'.gz')
if CLF == 'nr':
    clf = CClassifierRejectThreshold.load(fname)
    out = count_svm_prototypes(clf._clf)
elif CLF == 'dnr':
    clf = CClassifierDNR.load(fname)
    out = {}
    # Count for layer classifiers
    for l in clf._layers:
        out[l] = count_svm_prototypes(clf._layer_clfs[l])
    # Count for combiner
    out['combiner'] = count_svm_prototypes(clf._clf)
else:
    raise NotImplementedError()

print(out)