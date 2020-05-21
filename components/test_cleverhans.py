import os
import numpy as np
from secml.adv.attacks import CAttackEvasionCleverhans
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.data.loader import CDataLoaderMNIST
from cleverhans.attacks import FastGradientMethod
from secml.ml.peval.metrics import CMetricAccuracy, CMetric

from mnist.mnist import mnist


def eval(clf, dset):
    X, y = dset.X, dset.Y
    # Predict
    y_pred = clf.predict(X)
    # Evaluate the accuracy of the classifier
    return CMetricAccuracy().performance_score(y, y_pred)


if __name__ == '__main__':
    from setGPU import setGPU
    setGPU(-1)

    random_state = 999

    # Prepare data
    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    ts = loader.load('testing')
    # Normalize the data
    tr.X /= 255
    ts.X /= 255

    # Get dnn
    dnn = mnist()
    if not os.path.exists("../mnist/mnist.pkl"):
        dnn.verbose = 1
        dnn.fit(tr.X, tr.Y)
        dnn.save_model("mnist.pkl")
    else:
        dnn.load_model("mnist.pkl")

    tr_acc = eval(dnn, tr)
    print("Accuracy on training set: {:.2%}".format(tr_acc))

    # Test
    cl_ts = ts[:1000, :]
    ts_acc = eval(dnn, ts)
    print("Accuracy on test set: {:.2%}".format(ts_acc))

    #  ======== Cleverhans ========
    dmax = 3.0  # Maximum perturbation
    lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    attack_cls = FastGradientMethod
    attack_params = {'eps': dmax,
                     'clip_max': ub,
                     'clip_min': lb,
                     'ord': np.inf}

    attack = CAttackEvasionCleverhans(
        classifier=dnn,
        surrogate_classifier=dnn,
        surrogate_data=tr,
        y_target=y_target,
        clvh_attack_class=attack_cls,
        **attack_params)

    print("Attack started...")
    attack_ds = ts[:100, :]
    eva_y_pred, _, eva_adv_ds, _ = attack.run(attack_ds.X, attack_ds.Y)
    print("Attack complete!")

    metric = CMetric.create('accuracy')
    acc = metric.performance_score(
        y_true=attack_ds.Y, y_pred=dnn.predict(attack_ds.X))
    acc_attack = metric.performance_score(
        y_true=attack_ds.Y, y_pred=eva_y_pred)

    print("Accuracy on reduced test set before attack: {:.2%}".format(acc))
    print("Accuracy on reduced test set after attack: {:.2%}".format(acc_attack))

    # ++++++++ SecEval ++++++++

    e_vals = CArray.arange(start=0., step=0.25, stop=1.1)

    # Run sec_eval
    sec_eval = CSecEval(attack=attack, param_name='attack_params.eps', param_values=e_vals)
    # Run the security evaluation using the test set
    print("Running security evaluation")
    sec_eval.verbose = 1
    sec_eval.run_sec_eval(attack_ds)  # , double_init=False)

    # Plot
    fig = CFigure(5, 12)
    fig.sp.plot_sec_eval(sec_eval.sec_eval_data, marker='o', show_average=True)
    fig.savefig('cleverhans_test.png')

    print("done?")



