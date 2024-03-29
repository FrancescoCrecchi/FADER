from secml.adv.attacks import CAttackEvasionPGDExp
from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit

from mnist.attack_dnn import security_evaluation

CLF = 'svm'
# TYPE = 'cat_hinge'
if __name__ == '__main__':
    seed = 999

    n_features = 2  # Number of features
    n_samples = 1250  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters
    ds = CDLRandomBlobs(n_features=n_features,
                        centers=centers,
                        cluster_std=cluster_std,
                        n_samples=n_samples,
                        random_state=seed).load()

    tr, ts = CTrainTestSplit(test_size=0.3, random_state=seed).split(ds)

    # nmz = CNormalizerMinMax()
    # tr.X = nmz.fit_transform(tr.X)
    # ts.X = nmz.transform(ts.X)

    # Load attack
    pgd_attack = CAttackEvasionPGDExp.load('{}_attack.gz'.format(CLF))
    # pgd_attack = CAttackEvasionPGDExp.load('{}_attack_{}.gz'.format(CLF, seed))
    # pgd_attack = CAttackEvasionPGDExp.load('{}_attack_{}.gz'.format(CLF, TYPE))

    # Run sec_eval
    eps = CArray.arange(start=0, step=0.1, stop=5+1e-5)
    sec_eval = security_evaluation(pgd_attack, ts, eps)

    # Dump to disk
    sec_eval.save('{}_sec_eval'.format(CLF))
    # sec_eval.save('{}_{}_sec_eval'.format(CLF, seed))
    # sec_eval.save('{}_{}_sec_eval'.format(CLF, TYPE))
