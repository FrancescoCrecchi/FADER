from secml.ml import CClassifier
from secml.ml.classifiers.reject import CClassifierReject


class CClassifierRejectSurrogate(CClassifierReject):

    def __init__(self, clf_rej, gamma_smoothing=1.0):
        self._clf_rej = clf_rej
        self._gamma_smooth = gamma_smoothing

        # internals
        self._caching = None
        super().__init__()

    @property
    def classes(self):
        return self._clf_rej.classes

    @property
    def n_classes(self):
        return self._clf_rej.n_classes

    @property
    def n_features(self):
        return self._clf_rej.n_features

    # IMPLEMENT

    def predict(self, x, return_decision_function=False, n_jobs=1):
        return self._clf_rej.predict(x, return_decision_function)

    # DO NOT FIT
    def _fit(self, x, y):
        pass

    # FWD/BWD OVERRIDE
    def forward(self, x, caching=True):
        self._caching = caching
        return super().forward(x, self._caching)

    def _forward(self, x):
        return self._clf_rej.forward(x, caching=self._caching)

    # def _backward(self, w):
    #     return self._clf_rej.gradient(self._cached_x, w)

    def _backward(self, w):
        # Due to gradient-masking, it can be useful flatten the support region for plateaus
        '''
        if grad.norm() < 0.01:
            - si salvano in variabili temporanee i gamma dei kernel degli SVM
            - si cambiano il gamma dei kernel degli SVM dividendoli (es. per 1000)
            - si somma al grad calcolato prima, il gradiente calcolato con i gamma modificati
            - si ripristinano i gamma originali
        '''
        grad = self._clf_rej.gradient(self._cached_x, w)

        if grad.norm() < 0.01:
            self.logger.info('** Smoothing Activated ***')
            orig_grad = grad.deepcopy()  # DEBUG

            # 1. Reduce gammas:
            # for c in range(self._clf_rej.n_classes - 1):
            #     self._clf_rej._clf._binary_classifiers[c].kernel.gamma /= self._gamma_smooth
            self._clf_rej._clf.kernel.gamma /= self._gamma_smooth

            # 2. Update computed gradient:
            grad += self._clf_rej.gradient(self._cached_x, w)

            # 3. Restore gammas:
            # for c in range(self._clf_rej.n_classes - 1):
            #     self._clf_rej._clf._binary_classifiers[c].kernel.gamma *= self._gamma_smooth
            self._clf_rej._clf.kernel.gamma *= self._gamma_smooth

            # DEBUG: DOUBLE CHECK
            restored_grad = self._clf_rej.gradient(self._cached_x, w)
            assert (orig_grad-restored_grad).norm() < 1e-8, "Something wrong here!"

        return grad

    @property
    def _grad_requires_forward(self):
        return self._clf_rej._grad_requires_forward


