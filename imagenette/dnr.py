from secml.array import CArray
from secml.ml import CKernelRBF, CNormalizerMeanStd, CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.classifiers.sklearn.c_classifier_svm import CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy

from torchvision.models import alexnet
import torch.nn as nn

from dataset_loading import load_imagenette, load_imagenet


N_TRAIN, N_TEST = 10000, 500
if __name__ == '__main__':
    random_state = 999

    vl = load_imagenette(exclude_val=True)
    ts = load_imagenet()

    # Load classifier
    net = alexnet(pretrained=True)
    linear = nn.Linear(in_features=4096, out_features=10, bias=True)
    linear.weight = nn.Parameter(
        net.classifier[-1].weight[
          [0, 217, 482, 491, 497, 566, 569, 571, 574, 701], :])
    linear.bias = nn.Parameter(
        net.classifier[-1].bias[
            [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]])
    net.classifier[-1] = linear
    dnn = CClassifierPyTorch(
        net, pretrained=True, input_shape=(3, 224, 224),
        preprocess=CNormalizerMeanStd(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create DNR
    layers = ['classifier:3', 'classifier:4', 'classifier:5']
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
    dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    '''
        Setting classifiers parameters
        IMAGENETTE		C	    gamma
        -----------------------------
        combiner	    1e-4	1
        classifier:3    1	    1e-4
        classifier:4    1	    1e-4
        classifier:5    1    	1e-4
        '''
    dnr.set_params({
        'classifier:3.C': 1,
        'classifier:3.kernel.gamma': 1e-4,
        'classifier:4.C': 1,
        'classifier:4.kernel.gamma': 1e-4,
        'classifier:5.C': 1,
        'classifier:5.kernel.gamma': 1e-4,
        'clf.C': 1e-4,
        'clf.kernel.gamma': 1
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN,
                                random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST,
                                random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Fit DNR
    dnr.verbose = 2  # DEBUG
    dnr.fit(tr_sample.X, tr_sample.Y)
    dnr.verbose = 0

    # Check test performance
    y_pred = dnr.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("DNR Accuracy: {}".format(acc))

    # Set threshold (FPR: 10%)
    dnr.threshold = dnr.compute_threshold(0.1, ts_sample)
    # Dump to disk
    dnr.save('dnr')
