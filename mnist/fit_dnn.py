from mnist.cnn_mnist import cnn_mnist_model


def get_datasets(random_state):
    from secml.data.loader import CDataLoaderMNIST

    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    ts = loader.load('testing')

    # Normalize
    tr.X /= 255.
    ts.X /= 255.

    # Select 30K samples to train DNN
    from secml.data.splitter import CTrainTestSplit
    tr, vl = CTrainTestSplit(train_size=30000, random_state=random_state).split(tr)

    return tr, vl, ts


if __name__ == '__main__':
    random_state = 999

    # Load data
    tr, vl, ts = get_datasets(random_state)

    # Fit DNN
    dnn = cnn_mnist_model()
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
    dnn.save_model('cnn_mnist.pkl')
