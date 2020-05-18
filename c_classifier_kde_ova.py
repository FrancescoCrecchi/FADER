from secml.data.loader import CDLRandom
from secml.data.splitter import CDataSplitterKFold
from secml.figure import CFigure
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.features import CNormalizerMinMax
from secml.ml.kernels import CKernelRBF

from c_classifier_kde import CClassifierKDE

if __name__ == '__main__':

    random_state = 999
    # Create test data
    dataset = CDLRandom(n_features=2,
                        n_redundant=0,
                        n_classes=4,
                        n_clusters_per_class=1,
                        random_state=random_state).load()
    dataset.X = CNormalizerMinMax().fit_transform(dataset.X)

    # Instantiate classifier
    rbf_kernel = CKernelRBF(gamma=1e1)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=rbf_kernel)

    # Validate model parameters indipendently
    clf.verbose = 1
    best_params = clf.estimate_parameters(dataset,
                                          parameters={'kernel.gamma': [1e-2, 1e-1, 1e0, 1e1, 1e2]},
                                          splitter='kfold',
                                          perf_evaluator='xval', #'xval-multiclass',
                                          metric='accuracy')

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])
    clf.fit(dataset.X, dataset.Y)

    # Test plot
    fig = CFigure()
    fig.sp.plot_ds(dataset)
    fig.sp.plot_decision_regions(clf)
    fig.title('KDE Classifier OVA')
    fig.savefig('c_classifier_kde_ova.png')

    print('done?')