from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.fit_dnn import get_datasets

CLF = 'dnr'
# CLF = 'hybrid_rbfnet_svm'
# CLF = 'dnr_rbf'

# Load data
random_state = 999
_, vl, ts = get_datasets(random_state)

# Load DNR
dnr = CClassifierDNR.load(CLF + '.gz')


# Check test performance
def evaluate_clf(clf):
    y_pred = clf.predict(ts.X, return_decision_function=False)
    return CMetricAccuracy().performance_score(ts.Y, y_pred)


print("*** LAYER CLFS ***")
for l in dnr._layers:
    acc = evaluate_clf(dnr._layer_clfs[l])
    print("- [{0}] Accuracy: {1}".format(l, acc))

print("*** COMBINER ***")
acc = evaluate_clf(dnr)
print("- Accuracy: {0}".format(acc))
