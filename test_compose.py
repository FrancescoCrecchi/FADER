from ptSNE import ptSNE
from c_classifier_kdebayes import CClassifierKDEBayes
from secml.ml.peval.metrics import CMetricAccuracy


def eval(clf, dset):
    X, y = dset.X, dset.Y
    # Predict
    y_pred = clf.predict(X)
    # Evaluate the accuracy of the classifier
    return CMetricAccuracy().performance_score(y, y_pred)


if __name__ == '__main__':
    import setGPU
    random_state = 999

    # Prepare data
    from secml.data.loader import CDataLoaderMNIST
    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    ts = loader.load('testing')
    # Normalize the data
    tr.X /= 255
    ts.X /= 255

    # Compose classifier
    tsne = ptSNE(tr.X[:3000, :], epochs=1000, d=2, verbose=1)
    clf = CClassifierKDEBayes(bandwidth='auto', preprocess=tsne)

    # Fit
    cl_tr = tr[3000:6000, :]
    clf.fit(cl_tr)

    tr_acc = eval(clf, cl_tr)
    print("Accuracy on training set: {:.2%}".format(tr_acc))

    # Test
    cl_ts = ts[:1000, :]
    ts_acc = eval(clf, cl_ts)
    print("Accuracy on test set: {:.2%}".format(ts_acc))

    print("done?")


