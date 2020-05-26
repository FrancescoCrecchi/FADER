from secml.array import CArray
from secml.ml.classifiers.reject import CClassifierRejectThreshold

from mnist.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, _, ts = get_datasets(random_state)

    # Select 1K test data (sampling)
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Load detector
    detector = CClassifierRejectThreshold.load('clf_rej.gz')

    # Perform white-box attack



