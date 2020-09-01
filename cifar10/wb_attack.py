import argparse

from secml.adv.attacks import CAttackEvasionPGDExp, CAttackEvasionPGD
from secml.array import CArray
from secml.core import CCreator
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy, CMetricAccuracyReject
from secml.utils import CLog

from cifar10.attack_dnn import security_evaluation
from cifar10.fit_dnn import get_datasets


class Callback(CCreator):

    def pre(self, seval):
        raise NotImplementedError()

    def post(self, seval):
        raise NotImplementedError()


class AttackParamsSchedulerCallback(Callback):

    def __init__(self, d):
        '''
        Schedule attack parameters according to a dictionary. E.g.:
        d = {
          1.0: params_dict
          2.0: params_dict
        }
        :param d: attack params dictionary mapping eps -> params
        '''
        self.d = d

    def pre(self, seval):
        eps = seval._attack.dmax
        if eps in self.d:
            seval._attack.set('solver_params', self.d[eps])

    def post(self, seval):
        pass


class MeasurePerformanceCallback(Callback):

    def __init__(self):
        self.acc_metric = CMetricAccuracy()
        self.rej_metric = CMetricAccuracyReject()
        # Internals
        self.performances = None
        self._c = 0

    def pre(self, seval):
        if self.performances is None:
            self.performances = CArray.zeros(shape=(seval.sec_eval_data.param_values.size,))

    def post(self, seval):
        f = None
        if seval._attack.dmax == 0:
            f = self.acc_metric.performance_score
        else:
            f = self.rej_metric.performance_score
        perf = f(seval.sec_eval_data.Y, seval.sec_eval_data.Y_pred[self._c])
        self.performances[self._c] = perf
        self._c += 1
        self.logger.info("dmax = {} -> accuracy: {:.2f}".format(seval._attack.dmax, perf))


EPS = CArray([0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0])
if __name__ == '__main__':

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("clf", help="Model type", type=str)
    parser.add_argument("-n", "--n_samples", help="Number of attack samples to use for security evaluation", type=int, default=100)
    parser.add_argument("-i", "--iter", help="Number of independent iterations to performs", type=int, default=3)
    parser.add_argument("-w", "--workers", help="Number of workers to use", type=int, default=1)
    args = parser.parse_args()

    random_state = 999
    tr, _, ts = get_datasets(random_state)

    print("- Attacking ", args.clf)

    # if args.clf == 'dnn':
    #     pgd_attack = CAttackEvasionPGD.load(args.clf + '_attack.gz')
    # else:
    # Load attack
    pgd_attack = CAttackEvasionPGDExp.load(args.clf + '_wb_attack.gz')

    # Setting up attack workers
    pgd_attack.n_jobs = args.workers

    # Check test performance
    clf = pgd_attack.surrogate_classifier
    y_pred = clf.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # HACK: Save attack parameters
    original_params = pgd_attack.solver_params.copy()

    for it in range(args.iter):

        print(" - It", str(it))
        # Select a sample of ts data
        it_idxs = CArray.randsample(ts.X.shape[0], shape=args.n_samples, random_state=random_state+it)
        ts_sample = ts[it_idxs, :]

        # Callbacks
        cbks = []
        if args.clf == 'dnr':
            # AttackParamsSchedulerCallback
            d = {
                1.0: {
                    'eta': 0.1,
                    'eta_min': 0.1,
                    'eta_pgd': 0.1,
                    'max_iter': 40,
                    'eps': 1e-10
                }
            }
            attack_scheduler_cbk = AttackParamsSchedulerCallback(d)
            cbks.append(attack_scheduler_cbk)

        # MeasurePerformanceCallback
        measure_perf_cbk = MeasurePerformanceCallback()
        measure_perf_cbk.verbose = 1
        cbks.append(measure_perf_cbk)

        # "Used to perturb all test samples"
        sec_eval = security_evaluation(pgd_attack,
                                       ts_sample,
                                       EPS,
                                       callbacks=cbks,
                                       save_adv_ds=True)

        # Save to disk
        sec_eval.save(args.clf + '_wb_seval_it_' + str(it))