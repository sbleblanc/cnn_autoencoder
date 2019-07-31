from torchtext.data import Dataset, Field, Example

def tokenize(text):
    return text.split(' ')


class AutoencodingDataset(Dataset):

    def __init__(self, dataset_fn, top_k, **kwargs):
        fields = [('text', Field(sequential=True, use_vocab=True, tokenize=tokenize))]
        examples = []
        counter = 0
        with open(dataset_fn, 'r') as in_file:
            for line in in_file:
                if counter >= top_k:
                    break
                examples.append(Example.fromlist([line.strip()], fields))
                counter += 1
        super(AutoencodingDataset, self).__init__(examples, fields, **kwargs)
        fields[0][1].build_vocab(self)
