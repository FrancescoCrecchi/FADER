from secml.data import CDataset
from secml.ml import CKernelRBF
from secml.ml.classifiers.sklearn.c_classifier_svm_m import CClassifierSVMM

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    # Load scores dset
    dset = CDataset.load('hybrid_scores_dset.gz')

    # Create clf
    clf = CClassifierSVMM(kernel=CKernelRBF(gamma=1), C=1)

    # Multiprocessing
    clf.n_jobs = 16

    # Xval
    xval_params = {'C': [1e-2, 1e-1, 1, 10, 100],
                   'kernel.gamma': [1e-3, 1e-2, 1e-1, 1]}

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Select and set the best training parameters for the classifier
    clf.verbose = 1
    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=dset,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy'
    )

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])