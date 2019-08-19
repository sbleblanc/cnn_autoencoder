import torch
import torch.nn as nn
import torch.optim as optim
from cnn_ae.utils.noise import noise_char_input


class WindowCorrectionTrainer(object):

    def __init__(self, model, optimizer, train_iter, test_iter, max_epoch, model_best_filename, model_end_filename, device):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model_best_filename = model_best_filename
        self.model_end_filename = model_end_filename
        self.device = device

    def train(self, end_epoch_callback=None):
        best = float('inf')
        for epoch in range(self.max_epoch):
            epoch_train_loss = 0.
            epoch_test_loss = 0.
            self.model.train()
            for batch in self.train_iter:
                self.optimizer.zero_grad()
                output = self.model(batch.noised)
                loss = self.criterion(output.view(-1, output.shape[-1]), batch.clean.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            self.model.eval()
            accuracy_elem_count = 0.
            accuracy_total_count = 0
            for batch in self.test_iter:
                output = self.model(batch.noised)
                loss = self.criterion(output.view(-1, output.shape[-1]), batch.clean.contiguous().view(-1))
                epoch_test_loss += loss.item()
                accuracy_elem_count += len(((output.argmax(dim=1) - batch.clean.squeeze(1)) == 0).nonzero())
                accuracy_total_count += batch.noised.shape[0]

            if end_epoch_callback:
                end_epoch_callback(self.model)

            epoch_train_loss = epoch_train_loss / len(self.train_iter)
            epoch_test_loss = epoch_test_loss / len(self.test_iter)
            epoch_test_accuracy = accuracy_elem_count / accuracy_total_count * 100

            if epoch_test_loss < best:
                best = epoch_test_loss
                torch.save(self.model.state_dict(), self.model_best_filename)
                print('Saved best model in {}'.format(self.model_best_filename))

            print('Epoch {}: Train={}, Test={}, Test Acc.={:.2f}%'.format(epoch + 1, epoch_train_loss, epoch_test_loss,
                                                                     epoch_test_accuracy))

        if self.model_end_filename:
            torch.save(self.model.state_dict(), self.model_end_filename)
            print('Saved last model in {}'.format(self.model_end_filename))
