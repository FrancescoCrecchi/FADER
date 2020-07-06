import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from toy import rbf_layer
# Generating a dataset for a given decision boundary
from toy.rbf_net import RBFNet

lims = (-2, 2)

x1 = np.linspace(lims[0], lims[1], 101)
x2 = 0.5 * np.cos(np.pi * x1) + 0.5 * np.cos(
    4 * np.pi * (x1 + 1))  # <- decision boundary

np.random.seed(0)
samples = 200
x = np.random.uniform(lims[0], lims[1], (samples, 2))
for i in range(samples):
    if i < samples // 2:
        x[i, 1] = np.random.uniform(lims[0], 0.5 * np.cos(
            np.pi * x[i, 0]) + 0.5 * np.cos(4 * np.pi * (x[i, 0] + 1)))
    else:
        x[i, 1] = np.random.uniform(
            0.5 * np.cos(np.pi * x[i, 0]) + 0.5 * np.cos(
                4 * np.pi * (x[i, 0] + 1)), lims[1])

tx = torch.from_numpy(x).float()
ty = torch.cat((torch.zeros(samples // 2, 1), torch.ones(samples // 2, 1)), dim=0)

# Instanciating and training an RBF network with the Gaussian basis function
# hidden representation with an RBF layer and then transforms that into a
# 1-dimensional output/prediction with a linear layer


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)


class MinMaxClipper:

    def __init__(self, min=0, max=1):
        self.min = min.item()
        self.max = max.item()

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'centres'):
            c = module.centres.data
            c.clamp_(self.min, self.max)


class RBFRBFNet(RBFNet):

    def __init__(self, layer_widths, layer_centres, basis_func):
        super(RBFRBFNet, self).__init__(layer_widths, layer_centres, basis_func)

    def fit(self, x, y, epochs, batch_size, lr, loss_func):
        self.train()
        obs = x.size(0)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        clipper = MinMaxClipper(min=x.min(), max=x.max())
        # self.apply(clipper)
        centers = [self.rbf_layers[0].centres.detach().numpy().copy()]
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)
                loss = loss_func(y_hat, y_batch.view(-1))
                current_loss += (1 / batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                sys.stdout.write(
                    '\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                    (epoch, progress, obs, current_loss))
                sys.stdout.flush()
            # self.apply(clipper)
            if (epoch < 500 and epoch % 5 == 0) or (epoch % 100 == 0):
                centers += [self.rbf_layers[0].centres.detach().numpy().copy()]
        return centers

# To add more layers, change the layer_widths and layer_centres lists

layer_widths = [2, 1]
layer_centres = [10]
# basis_func = rbf_layer.gaussian
basis_func = rbf_layer.gaussian_nopow
# basis_func = rbf_layer.linear

rbfnet = RBFRBFNet(layer_widths, layer_centres, basis_func)
centers_eval = rbfnet.fit(tx, ty, epochs=10000,
                          batch_size=samples, lr=0.05,
                          loss_func=nn.BCEWithLogitsLoss())
rbfnet.eval()

# Plotting the ideal and learned decision boundaries

steps = 500
x_span = np.linspace(lims[0], lims[1], steps)
y_span = np.linspace(lims[0], lims[1], steps)
xx, yy = np.meshgrid(x_span, y_span)
values = np.append(xx.ravel().reshape(xx.ravel().shape[0], 1),
                   yy.ravel().reshape(yy.ravel().shape[0], 1),
                   axis=1)

with torch.no_grad():
    preds = (
        torch.sigmoid(rbfnet(torch.from_numpy(values).float()))).data.numpy()

# centers = rbfnet.rbf_layers[0].centres.detach().numpy()
# print(centers)

fig, ax = plt.subplots(figsize=(4, 4), nrows=1, ncols=1)
ch = ax.contour(xx, yy, preds.reshape(xx.shape), levels=[0.5])
ax.scatter(x[:samples // 2, 0], x[:samples // 2, 1], s=20, c='dodgerblue', marker='+')
ax.scatter(x[samples // 2:, 0], x[samples // 2:, 1], s=20, c='orange', marker='x')

# centers_start = centers_eval[0]
# ax.scatter(centers_start[:, 0], centers_start[:, 1],
#            c=list(range(layer_centres[0])), s=50, marker='D', edgecolors='k', zorder=10)
#
# for c in centers_eval[1:-1]:
#     ax.scatter(c[:, 0], c[:, 1], s=25, c=list(range(layer_centres[0])), marker='o', edgecolors='k')

centers_final = centers_eval[-1]
print(centers_final)
ax.scatter(centers_final[:, 0], centers_final[:, 1],
           c=list(range(layer_centres[0])), s=100, marker='*', edgecolors='k', zorder=10)

ax.set_xlim(lims[0]*1.05, lims[1]*1.05)
ax.set_ylim(lims[0]*1.05, lims[1]*1.05)
ax.set_title('{:} ({:} basis)'.format(basis_func.__name__, layer_centres[0]))

fig.tight_layout()
fig.savefig("{:}_{:}.png".format(basis_func.__name__, layer_centres[0]))
