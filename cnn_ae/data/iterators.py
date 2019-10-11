import torch
from torchtext.data.iterator import Iterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
from torchtext.datasets.language_modeling import LanguageModelingDataset
from cnn_ae.data.datasets import RandomizedTextWindowDataset
from cnn_ae.common.exceptions import VocabNotBuiltException
from cnn_ae.utils.noise import binary_noise_char_input
import random as rnd

class NoisedPreWindowedIterator(Iterator):

    def __init__(self, dataset: RandomizedTextWindowDataset, batch_size: int, window_size: int, noise_ratio: float, **kwargs):
        if not hasattr(dataset.fields['text'], 'vocab'):
            raise VocabNotBuiltException("Must call build_vocab on the field passed to the dataset.")
        self.window_size = window_size
        self.noise_ratio = noise_ratio
        super(NoisedPreWindowedIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return len(self.dataset) // self.batch_size + (1 if len(self.dataset) % self.batch_size > 0 else 0)

    def __iter__(self):
        valid_noise = torch.tensor(list(range(2, len(self.dataset.fields['text'].vocab))), dtype=torch.long,
                                   device=self.device)
        text_field = self.dataset.fields['text']
        dataset = Dataset(examples=self.dataset.examples, fields=[('noised', text_field), ('clean', text_field)])
        if self.shuffle:
            shuffled_examples = [self.dataset.examples[i] for i in torch.randperm(len(self.dataset))]
        else:
            shuffled_examples = self.dataset.examples
        for b in range(len(self)):
            actual_batch_size = min(self.batch_size, len(self.dataset) - b*self.batch_size)
            batch_clean = torch.zeros([actual_batch_size, self.window_size], dtype=torch.long, device=self.device)
            for i in range(actual_batch_size):
                example_index = b * self.batch_size + i
                batch_clean[i] = text_field.numericalize(shuffled_examples[example_index].text).squeeze(1)
            if self.noise_ratio > 0:
                batch_noised = binary_noise_char_input(batch_clean, valid_noise, self.noise_ratio)
            else:
                batch_noised = batch_clean
            yield Batch.fromvars(dataset, actual_batch_size, noised=batch_noised, clean=batch_clean)



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

    def __generate_text_data(self):
        text_field = self.dataset.fields['text']
        raw_text = self.dataset[0].text
        text_data = text_field.numericalize([raw_text], device=self.device).squeeze(0)
        valid_noise = torch.tensor(list(range(2, len(self.dataset.fields['text'].vocab))), dtype=torch.long,
                                   device=self.device)
        dataset = Dataset(examples=self.dataset.examples, fields=[('noised', text_field), ('clean', text_field)])
        return text_data, valid_noise, dataset

    def __generate_batch(self, batch, data_length, text_data, valid_noise):
        start = batch * self.batch_size
        end = start + min(self.batch_size, data_length - start - self.window_size)
        batch_clean = torch.zeros([end - start, self.window_size], dtype=torch.long, device=self.device)
        for bi, ti in enumerate(range(start, end)):
            batch_clean[bi] = text_data[ti:ti + self.window_size]
        if self.noise_ratio > 0:
            batch_noised = binary_noise_char_input(batch_clean, valid_noise, self.noise_ratio)
        else:
            batch_noised = batch_clean
        return start , end , batch_noised, batch_clean

    def __iter__(self):
        text_data, valid_noise, dataset = self.__generate_text_data()
        for b in range(len(self)):
            start, end, batch_noised, batch_clean = self.__generate_batch(b, len(self.dataset[0].text), text_data, valid_noise)
            yield Batch.fromvars(dataset, end - start, noised=batch_noised, clean=batch_clean)
    # def __iter__(self):
    #     text_field = self.dataset.fields['text']
    #     raw_text = self.dataset[0].text
    #     # if len(raw_text) % self.batch_size:
    #     #     diff_to_add = self.batch_size - (len(raw_text) % self.batch_size)
    #     #     raw_text.extend([text_field.pad_token for _ in range(diff_to_add)])
    #     text_data = text_field.numericalize([raw_text], device=self.device).squeeze(0)
    #     valid_noise = torch.tensor(list(range(2, len(self.dataset.fields['text'].vocab))), dtype=torch.long,
    #                                device=self.device)
    #     dataset = Dataset(examples=self.dataset.examples, fields=[('noised', text_field), ('clean', text_field)])
    #     for b in range(len(self)):
    #         start = b * self.batch_size
    #         end = start + min(self.batch_size, len(raw_text) - start - self.window_size)
    #         batch_clean = torch.zeros([end - start, self.window_size], dtype=torch.long, device=self.device)
    #         for bi, ti in enumerate(range(start, end)):
    #             batch_clean[bi] = text_data[ti:ti+self.window_size]
    #         if self.noise_ratio > 0:
    #             batch_noised = binary_noise_char_input(batch_clean, valid_noise, self.noise_ratio)
    #         else:
    #             batch_noised = batch_clean
    #         yield Batch.fromvars(dataset, end - start, noised=batch_noised, clean=batch_clean)

    def get_single_rnd_batch(self):
        text_data, valid_noise, dataset = self.__generate_text_data()
        b = rnd.randint(0, len(self))
        start, end, batch_noised, batch_clean = self.__generate_batch(b, len(self.dataset[0].text), text_data,
                                                                      valid_noise)
        return Batch.fromvars(dataset, end - start, noised=batch_noised, clean=batch_clean)


class PredictMiddleNoisedWindowIterator(Iterator):

    def __init__(self, noised_win_iter: Iterator, middle_width: int, **kwargs):
        super(PredictMiddleNoisedWindowIterator, self).__init__(noised_win_iter.dataset, noised_win_iter.batch_size, **kwargs)
        self.noised_win_iter = noised_win_iter
        self.middle_width = middle_width

    def __compute_middle_range(self):
        mid = self.noised_win_iter.window_size // 2
        middle_start = mid - (self.middle_width // 2)
        middle_end = middle_start + self.middle_width
        return middle_start, middle_end

    def __hole_batch(self, batch, middle_start, middle_end):
        holed_noised = torch.cat(
            [batch.noised[:, 0:middle_start], batch.noised[:, middle_end:self.noised_win_iter.window_size]],
            dim=1
        )
        clean_middle = batch.clean[:, middle_start:middle_end]
        return holed_noised, clean_middle

    def __iter__(self):
        middle_start, middle_end = self.__compute_middle_range()
        for b in self.noised_win_iter:
            holed_noised, clean_middle = self.__hole_batch(b, middle_start, middle_end)
            yield Batch.fromvars(b.dataset, b.batch_size, noised=holed_noised, clean=clean_middle)

    def get_single_rnd_batch(self):
        b = self.noised_win_iter.get_single_rnd_batch()
        middle_start, middle_end = self.__compute_middle_range()
        holed_noised, clean_middle = self.__hole_batch(b, middle_start, middle_end)
        return Batch.fromvars(b.dataset, b.batch_size, noised=holed_noised, clean=clean_middle)

    # def __iter__(self):
    #     mid = self.window_size // 2
    #     middle_start = mid - (self.middle_width // 2)
    #     middle_end = middle_start + self.middle_width
    #     for b in super(PredictMiddleNoisedWindowIterator, self).__iter__():
    #         holed_noised = torch.cat(
    #             [b.noised[:, 0:middle_start], b.noised[:, middle_end:self.window_size]],
    #             dim=1
    #         )
    #         clean_middle = b.clean[:, middle_start:middle_end]
    #         yield Batch.fromvars(b.dataset, b.batch_size, noised=holed_noised, clean=clean_middle)
