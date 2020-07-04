import torch
from torch import nn

from components.rbf_network import RBFNetwork


class Stack(nn.Module):

    def __init__(self):
        super(Stack, self).__init__()

    def forward(self, iterable, axis):
        x = torch.stack(iterable, axis)
        return x


class DeepRBFNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_classes):
        '''
        Init DeepRBFNetwork module
        :param n_features: list of input features
        :param n_hiddens: list layer neurons
        :param n_classes: dataset classes
        '''
        super(DeepRBFNetwork, self).__init__()
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_classes = n_classes
        # Internals
        self._n_layers = len(self.n_features)

        # Layer_clfs
        self._layer_clfs = nn.ModuleList()
        for i in range(self._n_layers):
            self._layer_clfs.append(RBFNetwork(self.n_features[i], self.n_hiddens[i], n_classes))
        self._stack = Stack()
        # # Set combiner on top
        # self._combiner = nn.ModuleList()
        # for _ in range(n_classes):
        #     # 'n_hiddens[-1]' combiner rbf neurons per class: RBFUnit + LinearUnit -> Class score
        #     rbfnet = RBFNetwork(self._n_layers, self.n_hiddens[-1], 1)
        #     self._combiner.append(rbfnet)
        self._combiner = nn.Linear(self.n_classes * self._n_layers, n_classes)

        # TODO: FIX BETAS?

    def forward(self, x):
        f_x = []
        # Unpack and distribute
        start = 0
        for i in range(self._n_layers):
            out = self._layer_clfs[i](x[:, start:start + self.n_features[i]])
            start += self.n_features[i]
            f_x.append(out)
        f_x = self._stack(f_x, 2)        # fx.shape=(batch_size, n_classes, n_layers)
        # # Pass through combiner x class
        # out = []
        # for c in range(self.n_classes):
        #     out.append(self._combiner[c](f_x[:, c, :]))
        # out = torch.cat(out, 1)
        out = self._combiner(f_x.view(x.shape[0], -1))
        return out