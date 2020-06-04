from secml.ml import CClassifier


class CClassifierDNRSurrogate(CClassifier):

    def __init__(self, dnr, gamma_smoothing=1.0):
        self._dnr = dnr
        self._gamma_smooth = gamma_smoothing
        super().__init__()

    @property
    def classes(self):
        return self._dnr.classes

    @property
    def n_classes(self):
        return self._dnr.n_classes

    @property
    def n_features(self):
        return self._dnr.n_features

    def _fit(self, x, y):
        pass

    def _forward(self, x):
        self._dnr._cached_x = self._cached_x
        return self._dnr._forward(x)

    def _backward(self, w):

        # Due to gradient-masking, it can be useful flatten the support region for plateaus
        '''
        if grad.norm() < 0.01:
            - si salvano in variabili temporanee i gamma dei kernel degli SVM
            - si cambiano il gamma dei kernel degli SVM dividendoli (es. per 1000)
            - si somma al grad calcolato prima, il gradiente calcolato con i gamma modificati
            - si ripristinano i gamma originali
        '''
        grad = self._dnr._backward(w)

        if grad.norm() < 0.01:
            orig_grad = grad.deepcopy()  # DEBUG

            # 1. Reduce gammas:
            # - Layer classifiers:
            for l in self._dnr._layers:
                # - Reduce kernel gammas
                for c in range(self._dnr.n_classes-1):
                    self._dnr._layer_clfs[l]._binary_classifiers[c].kernel.gamma /= self._gamma_smooth
            # - Collector
            for c in range(self._dnr.n_classes-1):
                self._dnr.clf._binary_classifiers[c].kernel.gamma /= self._gamma_smooth

            # 2. Update computed gradient:
            grad += self._dnr._backward(w)

            # 3. Restore gammas:
            # - Layer classifiers:
            for l in self._dnr._layers:
                for c in range(self._dnr.n_classes-1):
                    self._dnr._layer_clfs[l]._binary_classifiers[c].kernel.gamma *= self._gamma_smooth
            # - Collector
            for c in range(self._dnr.n_classes-1):
                self._dnr.clf._binary_classifiers[c].kernel.gamma *= self._gamma_smooth

            # DEBUG: DOUBLE CHECK
            restored_grad = self._dnr._backward(w)
            assert (orig_grad-restored_grad).norm() < 1e-8, "Something wrong here!"

        return grad

    @property
    def _grad_requires_forward(self):
        return self._dnr._grad_requires_forward


