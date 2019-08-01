import torch
import torch.nn as nn
import torch.optim as optim
from cnn_ae.utils.noise import noise_char_input


class DenoisingCNN(object):

    def __init__(self, model, optimizer, pad_idx, train_iter, test_iter, max_epoch, model_filename, device):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model_filename = model_filename
        self.device = device

    def train(self, end_epoch_callback=None):
        best = float('-inf')
        for epoch in range(self.max_epoch):
            epoch_train_loss = 0.
            epoch_test_loss = 0.
            self.model.train()
            for batch in self.train_iter:
                self.optimizer.zero_grad()
                input_data = batch.text.permute(1, 0)
                noised_input_data = noise_char_input(input_data, 0.0)
                output = self.model(noised_input_data)
                loss = self.criterion(output.view(-1, output.shape[-1]), input_data.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            self.model.eval()
            for batch in self.test_iter:
                input_data = batch.text.permute(1, 0)
                output = self.model(input_data)
                loss = self.criterion(output.view(-1, output.shape[-1]), input_data.contiguous().view(-1))
                epoch_test_loss += loss.item()

            if end_epoch_callback:
                end_epoch_callback(self.model)

            epoch_train_loss = epoch_train_loss/len(self.train_iter)
            epoch_test_loss = epoch_test_loss / len(self.test_iter)

            if epoch_test_loss > best:
                best = epoch_test_loss
                torch.save(self.model.state_dict(), self.model_filename)
                print('Saved best model in {}'.format(self.model_filename))

            print('Epoch {}: Train={}, Test={}'.format(epoch + 1, epoch_train_loss, epoch_test_loss))
