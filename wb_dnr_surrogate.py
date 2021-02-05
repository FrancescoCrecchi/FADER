from wb_nr_surrogate import CClassifierRejectSurrogate


class CClassifierDNRSurrogate(CClassifierRejectSurrogate):

    def __init__(self, clf_rej, gamma_smoothing=1.0):
        super().__init__(clf_rej, gamma_smoothing)

    def _backward(self, w):

        # Due to gradient-masking, it can be useful flatten the support region for plateaus
        '''
        if grad.norm() < 0.01:
            - si salvano in variabili temporanee i gamma dei kernel degli SVM
            - si cambiano il gamma dei kernel degli SVM dividendoli (es. per 1000)
            - si somma al grad calcolato prima, il gradiente calcolato con i gamma modificati
            - si ripristinano i gamma originali
        '''
        # self._clf_rej._cached_x = self._cached_x
        # grad = self._clf_rej.backward(w)
        grad = self._clf_rej.gradient(self._cached_x, w)

        if grad.norm() < 0.01:
            self.logger.debug('** Smoothing Activated ***')
            orig_grad = grad.deepcopy()  # DEBUG

            # 1. Reduce gammas:
            # - Layer classifiers:
            for l in self._clf_rej._layers:
                # - Reduce kernel gamma
                self._clf_rej._layer_clfs[l].kernel.gamma /= self._gamma_smooth
            # - Collector
            self._clf_rej._clf.kernel.gamma /= self._gamma_smooth

            # 2. Update computed gradient:
            grad = self._clf_rej.gradient(self._cached_x, w)

            # 3. Restore gammas:
            # - Layer classifiers:
            for l in self._clf_rej._layers:
                # - Reduce kernel gamma
                self._clf_rej._layer_clfs[l].kernel.gamma *= self._gamma_smooth
            # - Collector
            self._clf_rej._clf.kernel.gamma *= self._gamma_smooth

            # DEBUG: DOUBLE CHECK
            restored_grad = self._clf_rej.gradient(self._cached_x, w)
            assert (orig_grad-restored_grad).norm() < 1e-8, "Something wrong here!"

        return grad
