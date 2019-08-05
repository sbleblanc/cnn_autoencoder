import torch
import torch.nn as nn
import torch.optim as optim
from cnn_ae.utils.noise import noise_char_input


class DenoisingCNN(object):

    def __init__(self, model, optimizer, train_iter, test_iter, max_epoch, model_best_filename, model_end_filename, device, pad_idx=None):
        self.model = model
        if pad_idx:
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model_best_filename = model_best_filename
        self.model_end_filename = model_end_filename
        self.device = device

    def train(self, noise_ratio=0.0, end_epoch_callback=None):
        best = float('inf')
        for epoch in range(self.max_epoch):
            epoch_train_loss = 0.
            epoch_test_loss = 0.
            self.model.train()
            for batch in self.train_iter:
                self.optimizer.zero_grad()
                input_data = batch.text.permute(1, 0)
                noised_input_data = noise_char_input(input_data, noise_ratio)
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

            epoch_train_loss = epoch_train_loss
            epoch_test_loss = epoch_test_loss

            if epoch_test_loss < best:
                best = epoch_test_loss
                torch.save(self.model.state_dict(), self.model_best_filename)
                print('Saved best model in {}'.format(self.model_best_filename))

            print('Epoch {}: Train={}, Test={}'.format(epoch + 1, epoch_train_loss, epoch_test_loss))

        if self.model_end_filename:
            torch.save(self.model.state_dict(), self.model_end_filename)
            print('Saved last model in {}'.format(self.model_end_filename))
