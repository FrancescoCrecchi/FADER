import torch
from torch import nn
from torch import optim


class Flatten(nn.Module):
    """Layer custom per reshape del tensore
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class MNISTCNN(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 3-classes dataset."""
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.flat = Flatten()
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        # 1st block
        x = torch.relu(self.conv1(x))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        # 2nd block
        x = torch.relu(self.conv3(x))
        x = torch.relu(torch.max_pool2d(self.conv4(x), 2))
        # Flatten
        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def mnist():
    # Random seed
    torch.manual_seed(0)

    net = MNISTCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=0.01, momentum=0.9)

    from secml.ml.classifiers import CClassifierPyTorch
    return CClassifierPyTorch(model=net,
                             loss=criterion,
                             optimizer=optimizer,
                             epochs=25,
                             batch_size=128,
                             input_shape=(1, 28, 28),
                             random_state=0)
