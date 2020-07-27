import torch
from torch import nn
from torch.optim import Adam

from secml.array import CArray
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml import CClassifierPyTorch
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.peval.metrics import CMetric
from secml.utils import fm

from toy.rbf_net import RBFNet

dataset = CDLRandomBlobs(n_features=2, n_samples=100, centers=3,
                         cluster_std=0.8, random_state=0).load()

layer_widths = [(2, 3)]
layer_centres = [3]
basis_func = 'gaussian'
# basis_func = 'gaussian_nopow'
# basis_func = 'linear'

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
rbfnet = RBFNet(layer_widths, layer_centres, basis_func)
c0 = dataset.X[[0, 5, -1], :].tondarray() + 1e-4  # To avoid nan when computing rbf distances
print(c0)
rbfnet.rbf_layers[0].centres.data = torch.from_numpy(c0).float()

clf_norej = CClassifierPyTorch(rbfnet,
                               optimizer=Adam(rbfnet.parameters(),
                                              lr=0.01,
                                              weight_decay=0),
                               loss=nn.CrossEntropyLoss(),
                               epochs=1000, batch_size=dataset.num_samples,
                               input_shape=(2,),
                               softmax_outputs=True,
                               random_state=0)
clf_norej.verbose = 1
clf_norej.fit(dataset.X, dataset.Y)
clf_norej.verbose = 0

centers_final = CArray(
    rbfnet.rbf_layers[0].centres.detach().cpu().numpy().copy())
print(centers_final)

print("n_SV: {:}/{:}".format(layer_centres[0], dataset.num_samples))

print("Testing classifier creation ")
clf = CClassifierRejectThreshold(clf_norej, threshold=0.8)
clf.threshold = clf.compute_threshold(0.1, dataset)

# Classification of another dataset
y_pred_reject, score_pred_reject = clf.predict(
    dataset.X, return_decision_function=True)
y_pred, score_pred = clf_norej.predict(
    dataset.X, return_decision_function=True)

# Compute the number of rejected samples
n_rej = (y_pred_reject == -1).sum()
print("Rejected samples: {:}".format(n_rej))

print("Real: \n{:}".format(dataset.Y))
print("Predicted: \n{:}".format(y_pred))
print("Predicted with reject: \n{:}".format(y_pred_reject))

acc = CMetric.create('accuracy').performance_score(y_pred, dataset.Y)
print("Accuracy no rejection: {:}".format(acc))

rej_acc = CMetric.create('accuracy').performance_score(
    y_pred_reject[y_pred_reject != -1],
    dataset.Y[y_pred_reject != -1])
print("Accuracy with rejection: {:}".format(rej_acc))

fig = CFigure(height=2.5, width=2.5, markersize=4, fontsize=9)
# Plot dataset points

# mark the rejected samples
y = clf.predict(dataset.X)

# plot the dataset
fig.sp.plot_ds(dataset, colors=['dodgerblue', 'r', 'limegreen'])
fig.sp.scatter(dataset.X[y == -1, 0], dataset.X[y == -1, 1],
               s=25, c='k', marker='o', edgecolors='k', linewidths=1,
               zorder=10)
fig.sp.scatter(centers_final[:, 0], centers_final[:, 1],
               c='w', s=25, marker='o', linewidths=1,
               edgecolors='k', zorder=10)

# Plot objective function
fig.sp.plot_fun(clf.predict,
                multipoint=True,
                grid_limits=dataset.get_bounds(),
                levels=[-1],
                levels_style='-',
                plot_background=False,
                n_grid_points=300)
fig.sp.xlim(-3, 3)
fig.sp.ylim(0, 6)
fig.sp.yticks([0, 2, 4, 6])
fig.sp.title('RBF RBFNet (reject, r={:}, th={:.2f})'.format(
    layer_centres[0], clf.threshold))
fig.sp.grid(linestyle='--')

fig.savefig(fm.join(fm.abspath(__file__), 'figures/nn_reject.png'))
fig.savefig(fm.join(fm.abspath(__file__), 'figures/nn_reject.pdf'))

fig_norej = CFigure(height=2.5, width=2.5, markersize=4, fontsize=9)
# Plot dataset points

# plot the dataset
fig_norej.sp.plot_ds(dataset, colors=['dodgerblue', 'r', 'limegreen'])

# for i in range(centers_start.shape[0]):
#     fig_norej.sp.plot_path(centers_start[i, :].append(centers_final[i, :], axis=0),
#                            start_style='o', start_facecolor='w',
#                            final_style='*', final_facecolor='k')

# fig_norej.sp.scatter(c0[:, 0], c0[:, 1], s=50, marker='D',
#                      edgecolors='k', zorder=10)
fig_norej.sp.scatter(centers_final[:, 0], centers_final[:, 1],
                     c='w', s=25, marker='o', linewidths=1,
                     edgecolors='k', zorder=10)

# Plot objective function
fig_norej.sp.plot_fun(clf_norej.predict,
                      multipoint=True,
                      grid_limits=dataset.get_bounds(),
                      levels=dataset.classes.tolist(),
                      levels_style='-',
                      plot_background=False,
                      n_grid_points=300)
fig_norej.sp.xlim(-3, 3)
fig_norej.sp.ylim(0, 6)
fig_norej.sp.yticks([0, 2, 4, 6])
fig_norej.sp.title('RBF RBFNet (no reject, r={:})'.format(layer_centres[0]))
fig_norej.sp.grid(linestyle='--')

fig_norej.savefig(fm.join(fm.abspath(__file__), 'figures/nn_noreject.png'))
fig_norej.savefig(fm.join(fm.abspath(__file__), 'figures/nn_noreject.pdf'))

x0, y0 = dataset.X[0, :], dataset.Y[0]
print(x0)

noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 3  # Maximum perturbation
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.5,
    'eta_min': 0.1,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-4
}

from secml.adv.attacks.evasion import CAttackEvasionPGDLS

pgd_ls_attack = CAttackEvasionPGDLS(
    classifier=clf_norej,
    surrogate_classifier=clf_norej,
    surrogate_data=dataset,
    # double_init_ds=dataset,
    distance=noise_type,
    lb=None,
    ub=None,
    dmax=dmax,
    solver_params=solver_params,
    y_target=y_target)

# Run the evasion attack on x0
y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(x0, y0)

print("Original x0 label: ", y0.item())
print("Adversarial example label (PGD-LS): ", y_pred_pgdls.item())

print("Number of classifier gradient evaluations: {:}"
      "".format(pgd_ls_attack.grad_eval))

fig_evasion_norej = CFigure(height=2.5, width=2.5, markersize=4, fontsize=9)

fig_evasion_norej.sp.plot_fun(clf_norej.predict,
                              multipoint=True,
                              grid_limits=dataset.get_bounds(),
                              levels=dataset.classes.tolist(),
                              levels_style='-',
                              plot_background=False,
                              n_grid_points=300)
# Convenience function for plotting the attack objective function
ch = fig_evasion_norej.sp.plot_fun(pgd_ls_attack.objective_function,
                                   plot_levels=False,
                                   grid_limits=dataset.get_bounds(),
                                   colorbar=False,
                                   vmin=-1.2, vmax=1.2,
                                   plot_background=True, n_colors=500,
                                   multipoint=True, n_grid_points=200)
fig_evasion_norej.sp.colorbar(ch, ticks=[-0.8, -0.4, 0, 0.4, 0.8])

fig_evasion_norej.sp.plot_ds(dataset, colors=['dodgerblue', 'r', 'limegreen'])

# Construct an array with the original point and the adversarial example
adv_path = x0.append(adv_ds_pgdls.X, axis=0)

# Convenience function for plotting the optimization sequence
fig_evasion_norej.sp.plot_path(adv_path, markersize=9)

fig_evasion_norej.sp.xlim(-3, 3)
fig_evasion_norej.sp.ylim(0, 6)
fig_evasion_norej.sp.yticks([0, 2, 4, 6])

fig_evasion_norej.sp.grid(grid_on=False)

fig_evasion_norej.sp.title(
    r"RBF RBFNet (no reject, $\varepsilon={:}$)".format(dmax))
fig_evasion_norej.savefig(
    fm.join('figures/nn_evasion_noreject.png'))
fig_evasion_norej.savefig(
    fm.join('figures/nn_evasion_noreject.pdf'))

x0, y0 = dataset.X[0, :], dataset.Y[0]
print(x0)

noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 3  # Maximum perturbation
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.5,
    'eta_min': 0.1,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-4
}

from secml.adv.attacks.evasion import CAttackEvasionPGDLS

pgd_ls_attack_reject = CAttackEvasionPGDLS(
    classifier=clf,
    double_init_ds=dataset,
    distance=noise_type,
    lb=None,
    ub=None,
    dmax=dmax,
    solver_params=solver_params,
    y_target=y_target)

# Run the evasion attack on x0
y_pred_pgdls_reject, _, adv_ds_pgdls_reject, _ = pgd_ls_attack_reject.run(x0,
                                                                          y0)

print("Original x0 label: ", y0.item())
print("Adversarial example label (PGD-LS): ", y_pred_pgdls_reject.item())

print("Number of classifier gradient evaluations: {:}"
      "".format(pgd_ls_attack_reject.grad_eval))

fig_evasion = CFigure(height=2.5, width=2.5, markersize=4, fontsize=9)

fig_evasion.sp.plot_fun(clf.predict,
                        multipoint=True,
                        grid_limits=dataset.get_bounds(),
                        levels=[-1],
                        levels_style='-',
                        plot_background=False,
                        n_grid_points=300)
# Convenience function for plotting the attack objective function
ch = fig_evasion.sp.plot_fun(pgd_ls_attack_reject.objective_function,
                             plot_levels=False,
                             grid_limits=dataset.get_bounds(),
                             plot_background=True, n_colors=100,
                             colorbar=False,
                             vmin=-1.2, vmax=1.2,
                             multipoint=True, n_grid_points=200)
fig_evasion.sp.colorbar(ch, ticks=[-0.8, -0.4, 0, 0.4, 0.8])

fig_evasion.sp.plot_ds(dataset, colors=['dodgerblue', 'r', 'limegreen'])
y = clf.predict(dataset.X)
fig_evasion.sp.scatter(dataset.X[y == -1, 0], dataset.X[y == -1, 1],
                       s=25, c='k', marker='o', edgecolors='k', linewidths=1,
                       zorder=10)

# Construct an array with the original point and the adversarial example
adv_path = x0.append(adv_ds_pgdls.X, axis=0)

# Convenience function for plotting the optimization sequence
fig_evasion.sp.plot_path(adv_path, markersize=9)

fig_evasion.sp.xlim(-3, 3)
fig_evasion.sp.ylim(0, 6)
fig_evasion.sp.yticks([0, 2, 4, 6])

fig_evasion.sp.grid(grid_on=False)

fig_evasion.sp.title(
    r"RBF RBFNet (reject, $\varepsilon={:}$)".format(dmax))
fig_evasion.savefig(
    fm.join(fm.abspath(__file__), 'figures/nn_evasion_reject.png'))
fig_evasion.savefig(
    fm.join(fm.abspath(__file__), 'figures/nn_evasion_reject.pdf'))
