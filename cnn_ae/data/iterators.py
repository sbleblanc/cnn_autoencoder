import torch
from torchtext.data.iterator import Iterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
from torchtext.datasets.language_modeling import LanguageModelingDataset
from cnn_ae.common.exceptions import VocabNotBuiltException
from cnn_ae.utils.noise import binary_noise_char_input


class NoisedWindowIterator(Iterator):

    def __init__(self, dataset: LanguageModelingDataset, batch_size: int, window_size: int, noise_ratio: float, **kwargs):
        if not hasattr(dataset.fields['text'], 'vocab'):
            raise VocabNotBuiltException("Must call build_vocab on the field passed to the dataset.")
        self.window_size = window_size
        self.noise_ratio = noise_ratio
        super(NoisedWindowIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        num_windows = len(self.dataset[0].text) - (self.window_size - 1)
        return num_windows // self.batch_size + num_windows % self.batch_size

    def __iter__(self):
        text_field = self.dataset.fields['text']
        text_data = text_field.numericalize([self.dataset[0].text], device=self.device).squeeze(0)
        valid_noise = torch.tensor(list(range(2, len(self.dataset.fields['text'].vocab))), dtype=torch.long)
        dataset = Dataset(examples=self.dataset.examples, fields=[('noised', text_field), ('clean', text_field)])
        for b in range(len(self)):
            start = b * self.batch_size
            end = min(start + self.batch_size, len(self))
            batch_clean = torch.zeros([end - start, self.window_size], dtype=torch.long)
            for bi, ti in enumerate(range(start, end)):
                batch_clean[bi] = text_data[ti:ti+self.window_size]
            batch_noised = binary_noise_char_input(batch_clean, valid_noise, self.noise_ratio)
            yield Batch.fromvars(dataset, end - start, noised=batch_noised, clean=batch_clean)
