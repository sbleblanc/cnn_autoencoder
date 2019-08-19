import torch
from torchtext.data.iterator import Iterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
from torchtext.datasets.language_modeling import LanguageModelingDataset
from cnn_ae.common.exceptions import VocabNotBuiltException
from cnn_ae.utils.noise import binary_noise_char_input


class NoisedWindowIterator(Iterator):

    def __init__(self, dataset: LanguageModelingDataset, batch_size: int, window_size: int, noise_ratio: float,
                 **kwargs):
        if not hasattr(dataset.fields['text'], 'vocab'):
            raise VocabNotBuiltException("Must call build_vocab on the field passed to the dataset.")
        self.window_size = window_size
        self.noise_ratio = noise_ratio
        super(NoisedWindowIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        num_windows = len(self.dataset[0].text) - (self.window_size - 1)
        return num_windows // self.batch_size + (1 if num_windows % self.batch_size else 0)

    def __iter__(self):
        text_field = self.dataset.fields['text']
        raw_text = self.dataset[0].text
        # if len(raw_text) % self.batch_size:
        #     diff_to_add = self.batch_size - (len(raw_text) % self.batch_size)
        #     raw_text.extend([text_field.pad_token for _ in range(diff_to_add)])
        text_data = text_field.numericalize([raw_text], device=self.device).squeeze(0)
        valid_noise = torch.tensor(list(range(2, len(self.dataset.fields['text'].vocab))), dtype=torch.long,
                                   device=self.device)
        dataset = Dataset(examples=self.dataset.examples, fields=[('noised', text_field), ('clean', text_field)])
        for b in range(len(self)):
            start = b * self.batch_size
            end = start + min(self.batch_size, len(raw_text) - start - self.window_size)
            batch_clean = torch.zeros([end - start, self.window_size], dtype=torch.long, device=self.device)
            for bi, ti in enumerate(range(start, end)):
                batch_clean[bi] = text_data[ti:ti+self.window_size]
            if self.noise_ratio > 0:
                batch_noised = binary_noise_char_input(batch_clean, valid_noise, self.noise_ratio)
            else:
                batch_noised = batch_clean
            yield Batch.fromvars(dataset, end - start, noised=batch_noised, clean=batch_clean)


class PredictMiddleNoisedWindowIterator(NoisedWindowIterator):

    def __init__(self, dataset: LanguageModelingDataset, batch_size: int, window_size: int, noise_ratio: float,
                 middle_width: int, **kwargs):
        super(PredictMiddleNoisedWindowIterator, self).__init__(dataset, batch_size, window_size, noise_ratio, **kwargs)
        self.middle_width = middle_width

    def __iter__(self):
        mid = self.window_size // 2
        middle_start = mid - (self.middle_width // 2)
        middle_end = middle_start + self.middle_width
        for b in super(PredictMiddleNoisedWindowIterator, self).__iter__():
            holed_noised = torch.cat(
                [b.noised[:, 0:middle_start], b.noised[:, middle_end:self.window_size]],
                dim=1
            )
            clean_middle = b.clean[:, middle_start:middle_end]
            yield Batch.fromvars(b.dataset, b.batch_size, noised=holed_noised, clean=clean_middle)
