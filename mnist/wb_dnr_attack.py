from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray

from mnist.attack_dnn import security_evaluation
from mnist.fit_dnn import get_datasets

N_TEST = 10         # TODO: restore this!
if __name__ == '__main__':
    random_state = 999

    _, _, ts = get_datasets(random_state)

    # Select 1K test data (sampling)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Load attack
    pgd_attack = CAttackEvasionPGDExp.load('dnr_wb_attack.gz')

    # "Used to perturb all test samples"
    eps = CArray.arange(start=0, step=0.5, stop=5.1)
    sec_eval = security_evaluation(pgd_attack, ts_sample, eps)

    # Save to disk
    sec_eval.save('dnr_wb_seval')



