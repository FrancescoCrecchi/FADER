from matplotlib.pyplot import ylabel
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
    get_accuracy_reject, plot_seceval_reject, plot_reject_percentage


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
    # print(c0)
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

    clf_norej.load_model(fm.join(fm.abspath(__file__), 'dnn_rbf_state.gz'))
    clf_norej.verbose = 0

    # HACK: Handle this in 'CClassifierReject'
    tmp = clf_norej.preprocess
    clf_norej.preprocess = None
    clf = CClassifierRejectThreshold(clf_norej, threshold=0.4, preprocess=tmp)

    clf.threshold = clf.compute_threshold(0.1, ds_ts)
    print(clf.threshold)

    get_accuracy_reject(clf, ds_ts)

    torch.cuda.empty_cache()  # TODO: TO FREE UP

    # Defining attack
    noise_type = 'l2'
    dmax = 2.1
    lb, ub = 0., 1.
    y_target = None

    eps = CArray([0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0])

    if False:

        # Should be chosen depending on the optimization problem
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'max_iter': 40,
            'eps': 1e-6
        }

        pgd_attack = CAttackEvasionPGDExp(classifier=clf,
                                          surrogate_classifier=clf,
                                          surrogate_data=ds_tr,
                                          # double_init=False,
                                          distance=noise_type,
                                          lb=lb, ub=ub,
                                          dmax=dmax,
                                          solver_params=solver_params,
                                          y_target=y_target)
        pgd_attack.verbose = 1
        # pgd_attack.n_jobs = 4

        # Attack sample
        N_ATTACK_POINTS = 100
        sample_idx = CArray.randsample(ds_ts.X.shape[0], shape=N_ATTACK_POINTS, random_state=random_state)
        ds_adv = ds_ts[sample_idx, :]

        # Security evaluation
        sec_eval = CSecEval(attack=pgd_attack,
                            param_name='dmax', param_values=eps,
                            save_adv_ds=False)
        sec_eval.verbose = 2  # DEBUG

        # Run the security evaluation using the test set
        print("Running security evaluation...")
        sec_eval.run_sec_eval(ds_adv) #, post_callbacks=[lambda seval: torch.cuda.empty_cache()])
        print("Done!")

        # Save to disk
        sec_eval.save(fm.join(fm.abspath(__file__), 'dnn_rbf_reject_seceval.gz'))

    sec_eval = CSecEval.load(
        fm.join(fm.abspath(__file__), 'dnn_rbf_reject_seceval.gz'))

    # Plots
    plot_seceval_reject(sec_eval, eps, label='DNN-RBF (R)', name='dnn_rbf')
    plot_reject_percentage(sec_eval, eps, label='DNN-RBF (R)', name='dnn_rbf')
