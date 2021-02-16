from secml.ml import CKernelRBF, CNormalizerMeanStd, CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.classifiers.sklearn.c_classifier_svm import CClassifierSVM
from secml.data.splitter import CDataSplitterKFold

from torchvision.models import alexnet

from dataset_loading import load_imagenette, load_imagenet


vl = load_imagenette(exclude_val=True)
ts = load_imagenet()

# Load classifier
net = alexnet(pretrained=True)
dnn = CClassifierPyTorch(
    net, pretrained=True, input_shape=(3, 224, 224),
    preprocess=CNormalizerMeanStd(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

# Create DNR
layers = ['classifier:3', 'classifier:4', 'classifier:5']
layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=1)
dnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
dnr.verbose = 1

C = [0.001, 0.01, 0.1, 1]
gamma = [1e-4, 1e-3, 1e-2, 1e-1]
splitter = CDataSplitterKFold(num_folds=2)
dnr.estimate_top_clf_parameters(vl, {"C": C, "kernel.gamma": gamma},
                                splitter, "accuracy")
