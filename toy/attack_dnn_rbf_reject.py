from secml.adv.attacks import CAttackEvasionPGDExp
from secml.adv.seceval import CSecEval
from secml.array import CArray
from secml.ml import CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.utils import fm

import torch
from torch import nn
from torch.optim import Adam

from toy.rbf_net import RBFNet
from toy.utils import get_cifar10_preprocess, get_datasets_cifar10, \
    get_accuracy_reject, plot_seceval_reject, rej_percentage


if __name__ == '__main__':
    random_state = 999

    # Load data
    ds_tr, ds_vl, ds_ts = get_datasets_cifar10(random_state)

    # CIFAR10 pre-trained Net Preprocessor
    dnn_pre = get_cifar10_preprocess()

    layer_widths = [(1024, 10)]
    layer_centres = [100]
    basis_func = 'gaussian'
    # basis_func = 'gaussian_nopow'
    # basis_func = 'linear'

    c0_idx = CArray.randsample(
        ds_vl.num_samples, shape=layer_centres[0], random_state=random_state)

    ds_vl_pre = dnn_pre.transform(ds_vl.X[c0_idx, :])

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    rbfnet = RBFNet(layer_widths, layer_centres, basis_func)
    c0 = ds_vl_pre.tondarray() + 1e-4  # To avoid nan when computing rbf distances
    print(c0)
    rbfnet.rbf_layers[0].centres.data = torch.from_numpy(c0).float()

    clf_norej = CClassifierPyTorch(rbfnet,
                                   optimizer=Adam(rbfnet.parameters(),
                                                  lr=0.01,
                                                  weight_decay=1e-6),
                                   loss=nn.CrossEntropyLoss(),
                                   epochs=200, batch_size=1000,
                                   input_shape=(1024,),
                                   softmax_outputs=False,
                                   random_state=0,
                                   preprocess=dnn_pre)

    clf_norej.load_state(fm.join(fm.abspath(__file__), 'dnn_rbf_state.gz'))
    clf_norej.verbose = 0

    clf = CClassifierRejectThreshold(clf_norej, threshold=0.4)

    # clf.threshold = clf.compute_threshold(0.01, ds_ts)
    print(clf.threshold)

    get_accuracy_reject(clf, ds_ts)

    torch.cuda.empty_cache()  # TODO: TO FREE UP

    # Defining attack
    noise_type = 'l2'
    dmax = 2.1
    lb, ub = 0., 1.
    y_target = None

    eps = CArray([0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0])

    if True:

        # Should be chosen depending on the optimization problem
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'max_iter': 40,
            'eps': 1e-6
        }

        pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                          double_init=False,
                                          distance=noise_type,
                                          lb=lb, ub=ub,
                                          dmax=dmax,
                                          solver_params=solver_params,
                                          y_target=y_target)
        pgd_attack.verbose = 1
        pgd_attack.n_jobs = 2

        # Attack sample
        sample_idx = CArray.randsample(
            ds_ts.X.shape[0], shape=25, random_state=random_state)
        ds_adv = ds_ts[sample_idx, :]

        # Security evaluation
        sec_eval = CSecEval(attack=pgd_attack,
                            param_name='dmax', param_values=eps,
                            save_adv_ds=False)
        sec_eval.verbose = 2  # DEBUG

        # Run the security evaluation using the test set
        print("Running security evaluation...")
        sec_eval.run_sec_eval(ds_adv)
        print("Done!")

        # Save to disk
        sec_eval.save(fm.join(fm.abspath(__file__), 'dnn_rbf_reject_seceval.gz'))

    sec_eval = CSecEval.load(
        fm.join(fm.abspath(__file__), 'dnn_rbf_reject_seceval.gz'))

    plot_seceval_reject(sec_eval, eps, label='DNN-RBF (R)', name='dnn_rbf')

    from secml.figure import CFigure
    fig = CFigure()
    fig.sp.plot(sec_eval.sec_eval_data.param_values,
                y=rej_percentage(sec_eval.sec_eval_data), marker='o')
    fig.show()
