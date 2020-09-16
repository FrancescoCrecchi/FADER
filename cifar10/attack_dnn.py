from secml.adv.attacks import CAttackEvasionPGD
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml import CNormalizerMeanStd
from secml.ml.peval.metrics import CMetricAccuracy

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets


def security_evaluation(attack, dset, evals,
                        callbacks=None,
                        # double_init=False,
                        save_adv_ds=False):

    # Security evaluation
    seval = CSecEval(attack=attack, param_name='dmax', param_values=evals, save_adv_ds=save_adv_ds)
    seval.verbose = 1  # DEBUG

    # Run the security evaluation using the test set
    print("Running security evaluation...")
    seval.run_sec_eval(dset,
                       # callbacks=callbacks,
                       # double_init=double_init
                       )
    print("Done!")

    return seval



N_SAMPLES = 100     # TODO: restore full dataset
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    # Load classifier
    dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    dnn.load_model('cnn_cifar10.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Load attack
    pgd_attack = CAttackEvasionPGD.load('dnn_attack.gz')

    # "Used to perturb all test samples"
    eps = CArray.arange(start=0, step=1/4, stop=2.1)
    sec_eval = security_evaluation(pgd_attack, ts[:N_SAMPLES, :], eps)

    # Save to disk
    sec_eval.save('dnn_seval')