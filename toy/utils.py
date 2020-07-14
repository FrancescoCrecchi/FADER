from secml.array import CArray
from secml.ml import CNormalizerMeanStd, CClassifierPyTorch, CNormalizerDNN
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject
from secml.figure import CFigure
from secml.utils import fm

from toy.cnn_cifar10 import cifar10


def get_datasets_cifar10(seed):
    # Load data
    from secml.data.loader import CDataLoaderCIFAR10
    tr, ts = CDataLoaderCIFAR10().load()

    # Select 40K samples to train DNN
    from secml.data.splitter import CTrainTestSplit
    tr, vl = CTrainTestSplit(train_size=40000, random_state=seed).split(tr)

    # Normalize
    tr.X /= 255.  # HACK: Done with Transforms
    vl.X /= 255.
    ts.X /= 255.

    return tr, vl, ts


def get_cifar10_preprocess():

    model = cifar10(pretrained=True)

    dnn = CClassifierPyTorch(model,
                             pretrained=True,
                             input_shape=(3, 32, 32),
                             preprocess=CNormalizerMeanStd(
                                 mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
                             random_state=0)

    dnn_pre = CNormalizerDNN(dnn, out_layer='features:24')

    return dnn_pre


def get_accuracy(clf, ds):

    y_pred = clf.predict(ds.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ds.Y, y_pred)
    print("Model Accuracy: {}".format(acc))


def get_accuracy_reject(clf, ds):

    y_pred = clf.predict(ds.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ds.Y, y_pred)
    acc_rej = CMetricAccuracyReject().performance_score(ds.Y, y_pred)
    print("Model Accuracy: {}".format(acc))
    print("Model Reject Accuracy: {}".format(acc_rej))


def plot_seceval(sec_eval, eps, label='', name=''):

    fig = CFigure(height=5, width=5)

    # Convenience function for plotting the Security Evaluation Curve
    fig.sp.plot_sec_eval(
        sec_eval.sec_eval_data, marker='o', label=label, show_average=True)
    fig.sp.xscale('symlog', linthreshx=0.1)
    fig.sp.xticks(eps)
    fig.sp.xticklabels(eps)
    fig.sp.ylim(-0.05, 1.05)
    fig.savefig(fm.join(fm.abspath(__file__), name + '_seceval_plot.pdf'))


def plot_seceval_reject(sec_eval, eps, label='', name=''):

    fig = CFigure(height=5, width=5)

    # Convenience function for plotting the Security Evaluation Curve
    fig.sp.plot_sec_eval(
        sec_eval.sec_eval_data, marker='o', label=label, show_average=True,
        metric=['accuracy'] + ['accuracy-reject'] * (sec_eval.sec_eval_data.param_values.size-1))
    fig.sp.xscale('symlog', linthreshx=0.1)
    fig.sp.xticks(eps)
    fig.sp.xticklabels(eps)
    fig.sp.ylim(-0.05, 1.05)
    fig.savefig(fm.join(fm.abspath(__file__), name + '_reject_seceval_plot.pdf'))


def rej_percentage(sec_eval_data):
    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):
        perf[k] = (sec_eval_data.Y_pred[k] == -1).sum() / sec_eval_data.Y_pred[k].shape[0]
    return perf
