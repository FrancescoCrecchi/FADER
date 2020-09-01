from secml.array import CArray
from secml.figure import CFigure
from mnist.rbf_net import CClassifierRejectRBFNet

FNAME = 'rbf_net_sigma_0.000_100.gz'
rbf_net = CClassifierRejectRBFNet.load(FNAME)
model = rbf_net._clf._clf.model     # RBFNetwork
linear = model.classifier           # Linear
w = list(linear.parameters())[0]    # Weights, shape: n_classes x n_features
n_hiddens = model.n_hiddens

n_classes, n_features = w.shape
fig = CFigure(height=5*n_classes, width=16)
for i in range(n_classes):
    print(i)
    x = CArray(w[i, :].abs().detach().cpu().numpy())

    # # Class features importance histogram
    # sp = fig.subplot(n_classes, 1, i+1)
    # sp.grid()
    # sp.bar(range(n_features), x)

    # # Box-and-whisker plot
    # l = []
    # start = 0
    # for h in n_hiddens:
    #     l.append(x[start:start+h])
    #     start += h
    # sp.boxplot(l)

    # Most important 10 features
    best_10_idx = x.argsort()[-10:][::-1]
    print(best_10_idx)
    print(x[best_10_idx])

fig.savefig("explain_rbfnet.png")