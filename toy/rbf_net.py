import torch.nn as nn

from toy import rbf_layer as rbf


class RBFNet(nn.Module):
    """RBF Network.

    The network consists on a set of RBF + Linear layers.

    Parameters
    ----------
    layer_widths : list of tuple
        For each set of RBF + Linear layers,
        a tuple of (in_features, out_features).
    layer_centres : list of int
        For each set of RBF + Linear layers,
        the number of centers of the RBF layer.
    basis_func : str
        The basis function, one from `rbf_layer.basis_func_dict`.
        Default 'gaussian'.

    """
    def __init__(self, layer_widths, layer_centres, basis_func='gaussian'):
        super(RBFNet, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths)):
            self.rbf_layers.append(
                rbf.RBF(layer_widths[i][0], layer_centres[i], basis_func))
            self.linear_layers.append(
                nn.Linear(layer_centres[i], layer_widths[i][1]))

    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out
