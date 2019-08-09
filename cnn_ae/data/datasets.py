import torch
from torchtext.data import Dataset, Field, Example

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

    def __string_to_tensor(self, s, n_pad):
        char_list = []
        for w in s.split(' '):
            char_list.extend(list(w))
            char_list.append('<S>')
        tensor_data = [self.fields['text'].vocab.stoi['<START>']]
        tensor_data.extend([self.fields['text'].vocab.stoi[c] for c in char_list[:-1]])
        tensor_data.append(self.fields['text'].vocab.stoi['<END>'])
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
