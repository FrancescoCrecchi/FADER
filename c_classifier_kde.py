"""
.. module:: ClassifierKernelDensityEstimator
   :synopsis: Kernel Density Estimator

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.figure import CFigure
from secml.ml.classifiers import CClassifier
from secml.ml.classifiers.clf_utils import \
    check_binary_labels, convert_binary_labels
from secml.ml.features import CNormalizerMinMax
from secml.ml.kernels import CKernel, CKernelRBF


class CClassifierKDE(CClassifier):
    """Kernel Density Estimator
    
    Parameters
    ----------
    kernel : None or CKernel subclass, optional
        Instance of a CKernel subclass to be used for computing
        similarity between patterns. If None (default), a linear
        SVM will be created.

    Attributes
    ----------
    class_type : 'kde'

    See Also
    --------
    .CKernel : Pairwise kernels and metrics.

    """

    __class_type = 'kde'

    def __init__(self, kernel=None, preprocess=None):

        # Calling CClassifier init
        super(CClassifierKDE, self).__init__(preprocess=preprocess)

        # Setting up the kernel function
        kernel_type = 'linear' if kernel is None else kernel
        self._kernel = CKernel.create(kernel_type)

        self._training_samples = None  # slot store training samples

    def __clear(self):
        self._training_samples = None

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._training_samples is None

    def is_linear(self):
        """Return True if the classifier is linear."""
        if (self.preprocess is None or self.preprocess is not None and
            self.preprocess.is_linear()) and self.is_kernel_linear():
            return True
        return False

    def is_kernel_linear(self):
        """Return True if the kernel is None or linear."""
        if self.kernel is None or self.kernel.class_type == 'linear':
            return True
        return False

    @property
    def kernel(self):
        """Kernel function (None if a linear classifier)."""
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value

    @property
    def training_samples(self):
        return self._training_samples

    @training_samples.setter
    def training_samples(self, value):
        self._training_samples = value

    def _fit(self, dataset):
        """Trains the One-Vs-All Kernel Density Estimator classifier.

        The following is a private method computing one single
        binary (2-classes) classifier of the OVA schema.

        Representation of each classifier attribute for the multiclass
        case is explained in corresponding property description.

        Parameters
        ----------
        dataset : CDataset
            Binary (2-class) training set. Must be a :class:`.CDataset`
            instance with patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifierKDE
            Instance of the KDE classifier trained using input dataset.

        """
        if dataset.num_classes > 2:
            raise ValueError("training can be performed on (1-classes) "
                             "or binary datasets only. If dataset is binary "
                             "only negative class are considered.")

        negative_samples_idx = dataset.Y.find(dataset.Y == 0)

        if negative_samples_idx is None:
            raise ValueError("training set must contain same negative samples")

        self._training_samples = dataset.X[negative_samples_idx, :]

        self.logger.info("Number of training samples: {:}"
                         "".format(self._training_samples.shape[0]))

        return self

    # def decision_function(self, x, y=1):
    #     """Computes the decision function for each pattern in x.
    #
    #     If a preprocess has been specified, input is normalized
    #      before computing the decision function.
    #
    #     Parameters
    #     ----------
    #     x : CArray
    #         Array with new patterns to classify, 2-Dimensional of shape
    #         (n_patterns, n_features).
    #     y : {0, 1}, optional
    #         The label of the class wrt the function should be calculated.
    #         Default is 1.
    #
    #     Returns
    #     -------
    #     score : CArray
    #         Value of the decision function for each test pattern.
    #         Dense flat array of shape (n_patterns,).
    #
    #     """
    #     if not self.is_fitted():
    #         raise ValueError("make sure the classifier is trained first.")
    #
    #     x = x.atleast_2d()  # Ensuring input is 2-D
    #
    #     # Preprocessing data if a preprocess is defined
    #     if self.preprocess is not None:
    #         x = self.preprocess.normalize(x)
    #
    #     return self._decision_function(x, y=y)

    def _forward(self, x):
        """Computes the decision function for each pattern in x.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_patterns,).

        """
        x = x.atleast_2d()  # Ensuring input is 2-D
        s = self.kernel.k(x, self._training_samples)
        s = CArray(s).mean(keepdims=False, axis=1)
        s = s.append(1.0 - s, axis=0).T     # Make it 2 classes
        return s

    def _backward(self, w):
        """Computes the gradient of the KDE classifier's decision function
        wrt decision function input.
        """
        k = self.kernel.gradient(self._cached_x)
        # Gradient sign depends on input label (0/1)
        return - convert_binary_labels(w.argmax()) * k.mean(axis=0)


if __name__ == '__main__':
    random_state = 999
    # Create test data
    dataset = CDLRandom(n_features=2,
                        n_redundant=0,
                        n_classes=2,
                        n_clusters_per_class=1,
                        random_state=random_state).load()
    dataset.X = CNormalizerMinMax().fit_transform(dataset.X)

    # Instantiate classifier
    kde = CClassifierKDE(kernel='rbf')
    # kde.kernel.gamma = 10

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Test estimate parameter
    kde.verbose = 1
    best_params = kde.estimate_parameters(dataset,
                                          parameters={'kernel.gamma': [1e-2, 1e-1, 1e0, 1e1, 1e2]},
                                          splitter=xval_splitter,
                                          metric='accuracy')
    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Retrain classifier
    kde.fit(dataset)

    # Predict
    s = kde.decision_function(dataset.X[:10, :])
    p = kde.predict(dataset.X[:10, :])

    # Test plot
    fig = CFigure()
    fig.sp.plot_ds(dataset)
    fig.sp.plot_decision_regions(kde)
    fig.title('kde Classifier')
    fig.savefig('c_classifier_kde.png')

    # Test gradient
    x = dataset.X[0, :]
    w = CArray.zeros(2, )
    w[1] = 1.
    grad = kde.gradient(x, w)
    print(grad)

    # Numerical gradient check
    from secml.ml.classifiers.tests.c_classifier_testcases import CClassifierTestCases

    CClassifierTestCases.setUpClass()
    CClassifierTestCases()._test_gradient_numerical(kde, dataset.X[10, :])

    print("done?")





