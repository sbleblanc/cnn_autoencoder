import torch
from torchtext.data import Dataset, Field, Example
from torchtext.datasets.language_modeling import LanguageModelingDataset

def tokenize(text):
    return text.split(' ')

class AutoencodingDataset(Dataset):

    def __init__(self, dataset_fn, top_k=None, min_len=7, add_init_eos=True, **kwargs):
        if add_init_eos:
            fields = [('text', Field(sequential=True, use_vocab=True, tokenize=tokenize, init_token='<START>',
                                     eos_token='<END>'))]
        else:
            fields = [('text', Field(sequential=True, use_vocab=True, tokenize=tokenize))]
        examples = []
        counter = 0
        with open(dataset_fn, 'r') as in_file:
            for line in in_file:
                if top_k and counter >= top_k:
                    break
                stripped = line.strip()
                if len(stripped) < min_len:
                    continue
                examples.append(Example.fromlist([stripped], fields))
                counter += 1
        super(AutoencodingDataset, self).__init__(examples, fields, **kwargs)
        fields[0][1].build_vocab(self)

    def __string_to_tensor_rnn(self, s, n_pad):
        char_list = []
        for w in s.split(' '):
            char_list.extend(list(w))
            char_list.append('<S>')
        tensor_data = [self.fields['text'].vocab.stoi['<START>']]
        tensor_data.extend([self.fields['text'].vocab.stoi[c] for c in char_list[:-1]])
        tensor_data.append(self.fields['text'].vocab.stoi['<END>'])
        tensor_data.extend([self.fields['text'].vocab.stoi['<pad>'] for _ in range(n_pad)])
        return torch.tensor(tensor_data).unsqueeze(0)

    def __string_to_tensor(self, s, n_pad):
        char_list = []
        for w in s.split(' '):
            char_list.extend(list(w))
            char_list.append('<S>')
        tensor_data = [self.fields['text'].vocab.stoi[c] for c in char_list[:-1]]
        tensor_data.extend([self.fields['text'].vocab.stoi['<pad>'] for _ in range(n_pad)])
        return torch.tensor(tensor_data).unsqueeze(0)

    def manual_strings_to_batch(self, *strings):
        tensors = []
        longest = 0
        for s in strings:
            if len(s) > longest:
                longest = len(s)
        for s in strings:
            tensors.append(self.__string_to_tensor(s, longest - len(s)))
        return torch.cat(tensors, dim=0)

    def batch_to_strings(self, batch):
        strings = []
        pad_idx = self.fields['text'].vocab.stoi['<pad>']
        for i in range(batch.shape[0]):
            chars = []
            for j in range(batch.shape[1]):
                if batch[i, j] == pad_idx:
                    continue
                chars.append(self.fields['text'].vocab.itos[batch[i, j]])
            strings.append(''.join(chars).replace('<S>', " "))
        return strings

class WindowExample(Example):

    def __init__(self, ref_str, win_start: int, win_end: int):
        self.ref_str = ref_str
        self.win_start = win_start
        self.win_end = win_end

    @property
    def text(self):
        return self.ref_str[self.win_start:self.win_end]


class RandomizedTextWindowDataset(Dataset):

    def __init__(self, path, text_field, window_size, newline_eos=True,
                 encoding='utf-8', topk=float('inf'), **kwargs):
        fields = [('text', text_field)]
        text = []
        with open(path, encoding=encoding) as f:
            line_counter = 0
            for line in f:
                text += text_field.preprocess(line)
                if newline_eos:
                    text.append(u'<eos>')
                line_counter += 1
                if line_counter >= topk:
                    break

        examples = []

        for i in range(len(text) - window_size):
            examples.append(WindowExample(text, i, i + window_size))

        super(RandomizedTextWindowDataset, self).__init__(examples, fields, **kwargs)


class SplittableLanguageModelingDataset(LanguageModelingDataset):

    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', topk=float('inf'), **kwargs):
        fields = [('text', text_field)]
        text = []
        with open(path, encoding=encoding) as f:
            line_counter = 0
            for line in f:
                text += text_field.preprocess(line)
                if newline_eos:
                    text.append(u'<eos>')
                line_counter += 1
                if line_counter >= topk:
                    break

        examples = [Example.fromlist([text], fields)]
        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        if stratified or random_state:
            raise NotImplemented()

        text = self.examples[0].text
        train_len = int(len(text) * split_ratio)
        fields = ('text', self.fields['text'])
        train_example = [Example.fromlist([text[0:train_len]], [fields])]
        test_example = [Example.fromlist([text[train_len:]], [fields])]

        return Dataset(train_example, self.fields), Dataset(test_example, self.fields)
