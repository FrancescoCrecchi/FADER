from cifar10.cnn_cifar10 import cifar10


def get_datasets(random_state):
    # Load data
    from secml.data.loader import CDataLoaderCIFAR10
    tr, ts = CDataLoaderCIFAR10().load()

    # Select 40K samples to train DNN
    from secml.data.splitter import CTrainTestSplit
    tr, vl = CTrainTestSplit(train_size=40000, random_state=random_state).split(tr)

    # # Normalize
    # tr.X /= 255.      # HACK: Done with Transforms
    vl.X /= 255.
    ts.X /= 255.

    return tr, vl, ts


if __name__ == '__main__':
    from secml.ml import CNormalizerMeanStd
    from secml.ml.peval.metrics import CMetricAccuracy
    from torchvision import transforms

    random_state = 999

    # Load data
    tr, vl, ts = get_datasets(random_state)

    # Transforms
    tr_transforms = transforms.Compose([
        transforms.Lambda(lambda x: x.reshape([3, 32, 32])),
        transforms.Lambda(lambda x: x.transpose([1, 2, 0])),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Fit DNN
    dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                  transform_train=tr_transforms)
    dnn.verbose = 1  # Can be used to display training process output

    print("Training started...")
    dnn.fit(tr.X, tr.Y)
    dnn.verbose = 0
    print("Training completed!")

    # Save to disk
    dnn.save_model('cnn_cifar10.pkl')

    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))
