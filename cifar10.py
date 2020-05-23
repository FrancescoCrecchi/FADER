"""Pretrained network from https://github.com/aaron-xichen/pytorch-playground


Modules for cifar10 net with n_channel=128:

features: Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (2): ReLU()
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (5): ReLU()
  (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (7): Dropout(p=0.1)
  (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (10): ReLU()
  (11): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (12): BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (13): ReLU()
  (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (15): Dropout(p=0.2)
  (16): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (17): BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (18): ReLU()
  (19): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (21): ReLU()
  (22): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (23): Dropout(p=0.30000000000000004)
  (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
  (25): BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True)
  (26): ReLU()
  (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (28): Dropout(p=0.4)
  (29): Flatten()
)
classifier: Sequential(
  (0): Linear(in_features=512, out_features=10, bias=True)
)

Layer output dimensions:
features:
    0 - 65536
    1 - 65536
    2 - 65536
    3 - 65536
    4 - 65536
    5 - 65536
    6 - 16384
    7 - 16384
    8 - 32768
    9 - 32768
    10 - 32768
    11 - 32768
    12 - 32768
    13 - 32768
    14 - 8192
    15 - 8192
    16 - 16384
    17 - 16384
    18 - 16384
    19 - 16384
    20 - 16384
    21 - 16384
    22 - 4096
    23 - 4096
    24 - 2048
    25 - 2048
    26 - 2048
    27 - 512
    28 - 512
    29 - 512
classifier:
    0 - 10
"""
import torch
from torch import nn, optim

from secml.ml.classifiers import CClassifierPyTorch


class Flatten(nn.Module):
    """Layer custom per reshape del tensore
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    p = 0.1
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Dropout(p)]
            p += 0.1
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False,
                                                  momentum=0.9),
                           nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def cifar10(lr=1e-2, momentum=0.9, weight_decay=1e-2, preprocess=None,
            softmax_outputs=False, random_state=None, epochs=75, gamma=0.1,
            batch_size=100, lr_schedule=(25, 50), n_channel=64):
    use_cuda = torch.cuda.is_available()
    if random_state is not None:
        torch.manual_seed(random_state)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M',
           4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    model.features = nn.Sequential(*model.features, Flatten())
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, gamma)
    return CClassifierPyTorch(model=model, loss=loss, optimizer=optimizer,
                              optimizer_scheduler=scheduler, epochs=epochs,
                              input_shape=(3, 32, 32), preprocess=preprocess,
                              random_state=random_state, batch_size=batch_size,
                              softmax_outputs=softmax_outputs)


if __name__ == "__main__":
    random_state = 999

    # Load data
    from secml.data.loader import CDataLoaderCIFAR10
    tr, ts = CDataLoaderCIFAR10().load()

    # Normalize
    tr.X /= 255.
    ts.X /= 255.

    # Select 40K samples to train DNN
    from secml.data.splitter import CTrainTestSplit
    tr, vl = CTrainTestSplit(train_size=40000, random_state=random_state).split(tr)

    # Fit DNN
    dnn = cifar10()
    dnn.verbose = 1  # Can be used to display training process output

    print("Training started...")
    dnn.fit(tr.X, tr.Y)
    dnn.verbose = 0
    print("Training completed!")

    y_pred = dnn.predict(ts.X, return_decision_function=False)
    from secml.ml.peval.metrics import CMetric

    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)

    print("Model Accuracy: {}".format(acc_torch))

    # Save to disk
    dnn.save_model('cifar10.pkl')
