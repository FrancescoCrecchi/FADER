import os

from secml.ml.classifiers.reject import CClassifierDNR

# PARAMETERS
DSET = 'cifar10'
CLF = 'dnr'


def count_svm_prototypes(svm):
    sv = abs(svm._alpha).sum(axis=0) > 0
    return sv.sum()


if CLF == 'dnr':
    clf = CClassifierDNR.load(os.path.join(DSET, CLF+'.gz'))
    out = {}
    # Count for layer classifiers
    for l in clf._layers:
        out[l] = count_svm_prototypes(clf._layer_clfs[l])
    # Count for combiner
    out['combiner'] = count_svm_prototypes(clf._clf)
else:
    raise NotImplementedError()

print(out)