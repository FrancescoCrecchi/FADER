import torch
from torch import nn

import torch_rbf.torch_rbf as rbf


class RBFNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_classes, beta=1.0):
        '''
        RBF Network PyTorch Module
        :param n_features: list of layer features sizes
        :param n_hiddens: list of hidden layer sizes
        :param n_classes: number of output classes
        '''
        super(RBFNetwork, self).__init__()
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        n_layers = len(self.n_features)
        # Internals
        self.rbf_layers = nn.ModuleList()
        # Make layers
        activation = lambda x: rbf.gaussian(x, beta=beta)
        for i in range(n_layers):
            self.rbf_layers.append(rbf.RBF(n_features[i], n_hiddens[i], activation))
            self.classifier = nn.Linear(sum(n_hiddens), n_classes)

    def forward(self, x):
        f_x = []
        # Compute layer embeddings through RBF units
        start = 0
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](x[:, start:start+self.n_features[i]])
            start += self.n_features[i]
            f_x.append(out)
        # Concatenate
        f_x = torch.cat(f_x, dim=1)
        # Feed through linear layer
        out = self.classifier(f_x)
        return out

def fit_model(model, x, y, epochs, batch_size, lr, loss_func):
    import sys
    from torch.utils.data import Dataset, DataLoader

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

    model.train()
    obs = x.size(0)
    trainset = MyDataset(x, y)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    while epoch < epochs:
        epoch += 1
        current_loss = 0
        batches = 0
        progress = 0
        for x_batch, y_batch in trainloader:
            batches += 1
            optimiser.zero_grad()
            y_hat = model.forward(x_batch)
            loss = loss_func(y_hat, y_batch)
            current_loss += (1/batches) * (loss.item() - current_loss)
            loss.backward()
            optimiser.step()
            progress += y_batch.size(0)
            sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                             (epoch, progress, obs, current_loss))
            sys.stdout.flush()

def test():
    import numpy as np
    import matplotlib.pyplot as plt

    # Generating a dataset for a given decision boundary
    x1 = np.linspace(-1, 1, 101)
    x2 = 0.5 * np.cos(np.pi * x1) + 0.5 * np.cos(4 * np.pi * (x1 + 1))  # <- decision boundary

    samples = 200
    x = np.random.uniform(-1, 1, (samples, 2))
    for i in range(samples):
        if i < samples // 2:
            x[i, 1] = np.random.uniform(-1, 0.5 * np.cos(np.pi * x[i, 0]) + 0.5 * np.cos(4 * np.pi * (x[i, 0] + 1)))
        else:
            x[i, 1] = np.random.uniform(0.5 * np.cos(np.pi * x[i, 0]) + 0.5 * np.cos(4 * np.pi * (x[i, 0] + 1)), 1)

    steps = 100
    x_span = np.linspace(-1, 1, steps)
    y_span = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    values = np.append(xx.ravel().reshape(xx.ravel().shape[0], 1),
                       yy.ravel().reshape(yy.ravel().shape[0], 1),
                       axis=1)

    tx = torch.from_numpy(x).float()
    ty = torch.cat((torch.zeros(samples // 2, 1), torch.ones(samples // 2, 1)), dim=0)

    # Instantiating and training an RBF network with the Gaussian basis function
    # This network receives a 2-dimensional input, transforms it into a 40-dimensional
    # hidden representation with an RBF layer and then transforms that into a
    # 1-dimensional output/prediction with a linear layer

    # To add more layers, change the layer_widths and layer_centres lists
    n_features = [2]
    n_hiddens = [40]
    n_classes = 1

    rbfnet = RBFNetwork(n_features, n_hiddens, n_classes, beta=1e-3)
    fit_model(rbfnet, tx, ty, 5000, samples, 0.01, nn.BCEWithLogitsLoss())
    rbfnet.eval()

    # Plotting the ideal and learned decision boundaries

    with torch.no_grad():
        preds = (torch.sigmoid(rbfnet(torch.from_numpy(values).float()))).data.numpy()
    ideal_0 = values[
        np.where(values[:, 1] <= 0.5 * np.cos(np.pi * values[:, 0]) + 0.5 * np.cos(4 * np.pi * (values[:, 0] + 1)))[0]]
    ideal_1 = values[
        np.where(values[:, 1] > 0.5 * np.cos(np.pi * values[:, 0]) + 0.5 * np.cos(4 * np.pi * (values[:, 0] + 1)))[0]]
    area_0 = values[np.where(preds[:, 0] <= 0.5)[0]]
    area_1 = values[np.where(preds[:, 0] > 0.5)[0]]

    fig, ax = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
    ax[0].scatter(x[:samples // 2, 0], x[:samples // 2, 1], c='dodgerblue')
    ax[0].scatter(x[samples // 2:, 0], x[samples // 2:, 1], c='orange', marker='x')
    ax[0].scatter(ideal_0[:, 0], ideal_0[:, 1], alpha=0.1, c='dodgerblue')
    ax[0].scatter(ideal_1[:, 0], ideal_1[:, 1], alpha=0.1, c='orange')
    ax[0].set_xlim([-1, 1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_title('Ideal Decision Boundary')
    ax[1].scatter(x[:samples // 2, 0], x[:samples // 2, 1], c='dodgerblue')
    ax[1].scatter(x[samples // 2:, 0], x[samples // 2:, 1], c='orange', marker='x')
    ax[1].scatter(area_0[:, 0], area_0[:, 1], alpha=0.1, c='dodgerblue')
    ax[1].scatter(area_1[:, 0], area_1[:, 1], alpha=0.1, c='orange')
    ax[1].set_xlim([-1, 1])
    ax[1].set_ylim([-1, 1])
    ax[1].set_title('RBF Decision Boundary')
    plt.show()

if __name__ == '__main__':
    # N_FEATURES = [128, 64, 32]
    # N_HIDDENS = [256, 128, 64]
    # N_CLASSES = 10
    #
    # # Instantiate object
    # model = RBFNetwork(N_FEATURES, N_HIDDENS, N_CLASSES)
    #
    # # Input
    # n_layers = len(N_FEATURES)
    # x = torch.randn(1, sum(N_FEATURES))
    #
    # # Fwd
    # output = model(x)
    # print(output)
    #
    # # Loss & Bwd
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output, torch.Tensor([8]).long())
    # print(loss)
    # loss.backward()

    test()

    print("done?")
