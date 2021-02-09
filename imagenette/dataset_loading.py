import torch
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import transforms

from secml.array import CArray
from secml.data import CDataset
from secml.utils import fm

import numpy as np

_imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def _flatten(x):
    """Flatten the dimension of the array that contains the features.
    """
    n_samples = x.shape[0]
    other_dims = x.shape[1:]
    n_features = CArray(other_dims).prod()
    x = x.reshape(n_samples, n_features)
    return x


def load_imagenet():
    if fm.file_exist("imagenet_val.gz"):
        return CDataset.load("imagenet_val.gz")
    else:
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        n_feat = 224 * 224 * 3
        imagenet_data = ImageNet('/home/asotgiu', split="val", transform=transform)
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=128,
                                                  shuffle=False, num_workers=10)
        X = CArray.zeros(shape=(500, n_feat))
        Y = CArray.zeros(shape=(500,))
        start = 0
        end = 0
        for i, data in enumerate(data_loader):
            labels = CArray(data[1]).astype(int)
            idxs = CArray(np.in1d(labels.tondarray(), _imagenette_classes))
            if idxs.nnz > 0:
                end += idxs.nnz
                X[start:end, :] = CArray(_flatten(data[0]))[idxs, :]
                Y[start:end] = labels[idxs]
                start = end
        for i, c in enumerate(_imagenette_classes):
            Y[Y == c] = i
        dataset = CDataset(X, Y.astype(int))
        dataset.save("imagenet_val.gz")
        return dataset


def load_imagenette(ds="all", exclude_val=False):
    if ds == "train_net":
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])
        n_feat = 256 * 256 * 3
    else:
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        n_feat = 224 * 224 * 3
    folder = "imagenette2-320"
    if exclude_val:
        folder += "-no-val"
    train = ImageFolder(f"../../{folder}/train", transform)
    val = ImageFolder(f"../../{folder}/val", transform)
    image_folder = torch.utils.data.ConcatDataset([train, val])
    data_loader = torch.utils.data.DataLoader(image_folder, batch_size=128,
                                              shuffle=False, num_workers=10)
    X = CArray.zeros(shape=(len(image_folder), n_feat))
    Y = CArray.zeros(shape=(len(image_folder))).astype(int)
    for i, data in enumerate(data_loader):
        X[i * 128: i * 128 + len(data[0]), :] = CArray(_flatten(data[0]))
        Y[i * 128: i * 128 + len(data[0])] = CArray(data[1]).astype(int)
    dataset = CDataset(X, Y)
    if ds == "all":
        return dataset
    idxs = []
    for i in range(dataset.num_classes):
        idx = CArray.randsample(CArray((dataset.Y == i).nnz_indices[1]),
                                shape=(dataset.Y == i).nnz, random_state=0)
        idxs.append(idx)
    if ds == "train_net":
        s_idx = []
        e_idx = []
        n_samples = 0
        for i in range(dataset.num_classes):
            n_samples += dataset.X[idxs[i][600:], :].shape[0]
            if i == 0:
                s_idx.append(0)
                e_idx.append(dataset.X[idxs[i][600:], :].shape[0])
            else:
                s_idx.append(e_idx[i - 1])
                e_idx.append(e_idx[i - 1] +
                             dataset.X[idxs[i][600:], :].shape[0])

        X = CArray.zeros(shape=(n_samples, n_feat))
        Y = CArray.zeros(shape=(n_samples,)).astype(int)
        for i in range(dataset.num_classes):
            # remaining samples for network training
            X[s_idx[i]:e_idx[i], :] = dataset.X[idxs[i][600:], :]
            Y[s_idx[i]:e_idx[i]] = dataset.Y[idxs[i][600:]]
        return CDataset(X, Y)
    elif ds == "test":
        X = CArray.zeros(shape=(2000, n_feat))
        Y = CArray.zeros(shape=(2000,)).astype(int)
        for i in range(dataset.num_classes):
            # 200 samples per class for testing
            X[i * 200: i * 200 + 200, :] = dataset.X[idxs[i][:200], :]
            Y[i * 200: i * 200 + 200] = dataset.Y[idxs[i][:200]]
        return CDataset(X, Y)
    elif ds == "train_det":
        X = CArray.zeros(shape=(4000, n_feat))
        Y = CArray.zeros(shape=(4000,)).astype(int)
        for i in range(dataset.num_classes):
            # 400 samples per class for detector training
            X[i * 400: i * 400 + 400, :] = dataset.X[idxs[i][200:600], :]
            Y[i * 400: i * 400 + 400] = dataset.Y[idxs[i][200:600]]
        return CDataset(X, Y)
