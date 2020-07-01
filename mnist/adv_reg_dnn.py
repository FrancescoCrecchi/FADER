import torch
from torch import nn, optim
from secml.ml import CClassifierPyTorch

from components.c_classifier_pytorch_rbf_network import grad_norm, gPenalty

from mnist.cnn_mnist import CNNMNIST
from mnist.fit_dnn import get_datasets
from mnist.rbf_net import plot_train_curves


class AdvNormRegClf(CClassifierPyTorch):

    def __init__(self, model, loss=None, optimizer=None, optimizer_scheduler=None, pretrained=False,
                 pretrained_classes=None, input_shape=None, random_state=None, preprocess=None, softmax_outputs=False,
                 epochs=10, batch_size=1, n_jobs=1, transform_train=None, validation_data=None, sigma=0.):
        self._sigma = sigma
        self._history = None

        super().__init__(model, loss, optimizer, optimizer_scheduler, pretrained, pretrained_classes, input_shape,
                         random_state, preprocess, softmax_outputs, epochs, batch_size, n_jobs, transform_train,
                         validation_data)

    def _fit(self, x, y):
        """Fit PyTorch model.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        """
        if any([self._optimizer is None,
                self._loss is None]):
            raise ValueError("Optimizer and loss should both be defined "
                             "in order to fit the model.")

        train_loader = self._data_loader(x, y, batch_size=self._batch_size,
                                         num_workers=self.n_jobs - 1,
                                         transform=self._transform_train)  # , shuffle=True)

        if self._validation_data:
            vali_loader = self._data_loader(self._validation_data.X,
                                            self._validation_data.Y,
                                            batch_size=self._batch_size,
                                            num_workers=self.n_jobs - 1)

        if self._history is None:  # FIRST RUN
            tr_loss, vl_loss = [], []
            xentr_loss, gnorm2, reg = [], [], []
        else:
            tr_loss, vl_loss = self._history['tr_loss'], self._history['vl_loss']
            xentr_loss, gnorm2, reg = self._history['xentr_loss'], self._history['grad_norm'], self._history['reg'],

        for epoch in range(self._epochs):
            train_loss = xentr = grad_norm2 = cum_penalty = 0.
            batches = 0
            for data in train_loader:
                batches += 1
                self._optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                # HACK: REGULARIZATION REQUIRES INPUT GRADIENT
                inputs.requires_grad = True
                outputs = self._model(inputs)
                loss = self._loss(outputs, labels)

                # Logging
                xentr += loss.item()
                grad_norm2 += grad_norm(loss, inputs).item()

                # HACK: Gradient norm regularization
                if self._sigma > 0:
                    penalty = gPenalty(inputs, loss, self._sigma, 2)
                    loss += penalty
                    cum_penalty += penalty.item()
                loss.backward()
                self._optimizer.step()
                # Accumulate loss
                train_loss += loss.item()

            # Mean across batches
            train_loss /= (batches + 1)
            xentr /= (batches + 1)
            grad_norm2 /= (batches + 1)
            cum_penalty /= (batches + 1)

            self.logger.debug(
                "[DEBUG] Epoch {} -> loss: {:.2e} (xentr:{:.3e}, grad_norm2:{:.3e}, penalty:{:.3e})".format(epoch + 1,
                                                                                                            train_loss,
                                                                                                            xentr,
                                                                                                            grad_norm2,
                                                                                                            cum_penalty))
            # print statistics
            if epoch % 10 == 0:
                if self._validation_data is not None:
                    # Compute validation performance
                    self._model.eval()  # enter evaluation mode
                    with torch.no_grad():
                        vali_loss = 0.0
                        vali_batches = 0
                        for data in vali_loader:
                            vali_batches += 1
                            inputs, labels = data
                            inputs = inputs.to(self._device)
                            labels = labels.to(self._device)
                            outputs = self._model(inputs)
                            loss = self._loss(outputs, labels)
                            vali_loss += loss.item()
                        # accumulate
                        vali_loss /= vali_batches

                    # Update curves
                    tr_loss.append(train_loss)
                    vl_loss.append(vali_loss)
                    xentr_loss.append(xentr)
                    gnorm2.append(grad_norm2)
                    reg.append(cum_penalty)

                    # Logging
                    self.logger.info(
                        '[epoch: %d] TR loss: %.3e - VL loss: %.3e' % (epoch + 1, tr_loss[-1], vl_loss[-1]))
                    self._model.train()  # restore training mode
                else:
                    # Update curves
                    tr_loss.append(train_loss)
                    # Logging
                    self.logger.info('[epoch: %d] TR loss: %.3f' % (epoch + 1, tr_loss[-1]))

            if self._optimizer_scheduler is not None:
                self._optimizer_scheduler.step()

        self._trained = True

        # HACK: Store training data for plots
        self._history = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'xentr_loss': xentr_loss,
            'grad_norm': gnorm2,
            'penalty': reg
        }

        return self._model


def adv_mnist_cnn(lr=0.1, momentum=0.9, weight_decay=0, preprocess=None,
                  softmax_outputs=False, random_state=None,
                  epochs=30, batch_size=128,
                  lr_schedule=(10, 20), gamma=0.1,
                  sigma=0., validation_data=None,
                  **kwargs):
    if random_state is not None:
        torch.manual_seed(random_state)
    model = CNNMNIST(**kwargs)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, gamma)
    return AdvNormRegClf(model=model, loss=loss, optimizer=optimizer,
                         optimizer_scheduler=scheduler, epochs=epochs,
                         input_shape=(1, 28, 28), preprocess=preprocess,
                         random_state=random_state, batch_size=batch_size,
                         softmax_outputs=softmax_outputs, sigma=sigma, validation_data=validation_data)


SIGMA = 0.
EPOCHS = 30
if __name__ == '__main__':
    random_state = 999

    # Load data
    tr, vl, ts = get_datasets(random_state)

    # Fit DNN
    dnn = adv_mnist_cnn(
        epochs=EPOCHS,
        sigma=SIGMA,
        validation_data=vl[:1000, :]
    )

    print("Training started...")
    dnn.verbose = 2  # Can be used to display training process output
    dnn.fit(tr.X, tr.Y)
    dnn.verbose = 0
    print("Training completed!")

    fig = plot_train_curves(dnn._history, SIGMA)
    fig.savefig('adv_reg_dnn_sigma_{:.2f}_curves.png'.format(SIGMA))

    from secml.ml.peval.metrics import CMetric

    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Save to disk
    dnn.save_model('adv_reg_dnn_sigma_{:.2f}.pkl'.format(SIGMA))
