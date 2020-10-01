import os

from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

DATASET = 'cifar10'
# CLF = 'dnr'
# CLF = 'hybrid_rbfnet_svm'
CLF = 'dnr_rbf'
# CLF = 'dnr_rbf_tr_init'

# Load data
random_state = 999

if DATASET == 'mnist':
    from mnist.fit_dnn import get_datasets
elif DATASET == 'cifar10':
    from cifar10.fit_dnn import get_datasets
else:
    ValueError("Unrecognized dataset!")

_, vl, ts = get_datasets(random_state)

# Load DNR
dnr = CClassifierDNR.load(os.path.join(DATASET, CLF + '.gz'))


# Check test performance
def evaluate_clf(clf):
    y_pred = clf.predict(ts.X, return_decision_function=False)
    return CMetricAccuracy().performance_score(ts.Y, y_pred)

print("========================")
print("Dataset: " + DATASET)
print("Classifier: " + CLF)
print("------------------------")


print("*** LAYER CLFS ***")
for l in dnr._layers:
    acc = evaluate_clf(dnr._layer_clfs[l])
    print("- [{0}] Accuracy: {1}".format(l, acc))

print("*** COMBINER ***")
# HACK TO AVOID REJECTION
dnr._threshold = -9999.
acc = evaluate_clf(dnr)
print("- Accuracy: {0}".format(acc))

print("========================")
