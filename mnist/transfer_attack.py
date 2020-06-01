from secml.adv.attacks import CAttackEvasionPGD
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold

from mnist.attack_dnn import security_evaluation
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets


N_SAMPLES = 1000       # TODO: restore full dataset
if __name__ == '__main__':
    random_state = 999
    tr, _, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Load clf_rej
    clf_rej = CClassifierRejectThreshold.load('clf_rej.gz')
    # Set threshold
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts)

    # Load attack and set params
    pgd_attack = CAttackEvasionPGD.load('dnn_attack.gz')
    pgd_attack._classifier = clf_rej
    pgd_attack.surrogate_classifier = dnn

    # "Used to perturb all test samples"
    eps = CArray.arange(start=0, step=0.5, stop=5.1)
    sec_eval = security_evaluation(pgd_attack, ts[:N_SAMPLES, :], eps)

    # Save to disk
    sec_eval.save('clf_rej_bb_seval_v2')
