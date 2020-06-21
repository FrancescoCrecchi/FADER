import torch
from torch import nn

import torch_rbf.torch_rbf as rbf


class RBFNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_classes):
        '''
        RBF Network PyTorch Module
        :param n_features: list of layer features sizes
        :param n_layers: number of layers
        :param n_hiddens:
        :param n_classes:
        '''
        super(RBFNetwork, self).__init__()
        self.n_features = n_features
        n_layers = len(self.n_features)
        # Internals
        self.rbf_layers = nn.ModuleList()
        # Make layers
        for i in range(n_layers):
            self.rbf_layers.append(rbf.RBF(n_features[i], n_hiddens, rbf.gaussian))
        self.classifier = nn.Linear(n_layers*n_hiddens, n_classes)

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


if __name__ == '__main__':
    N_FEATURES = [128, 64, 32]
    N_HIDDENS = 100
    N_CLASSES = 10

    # Instantiate object
    model = RBFNetwork(N_FEATURES, N_HIDDENS, N_CLASSES)

    # Input
    n_layers = len(N_FEATURES)
    x = torch.randn(1, sum(N_FEATURES))

    # Fwd
    output = model(x)
    print(output)

    # Loss & Bwd
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, torch.Tensor([8]).long())
    print(loss)
    loss.backward()

    print("done?")