from cnn_ae.data.iterators import PredictMiddleNoisedWindowIterator

class ManualTestingCallback(object):

    def __init__(self, dataset, test_strings, device):
        self.dataset = dataset
        self.test_strings = test_strings
        self.test_tensor = dataset.manual_strings_to_batch(*test_strings).to(device)

    def __call__(self, model):
        output = model(self.test_tensor).softmax(dim=2).argmax(dim=2)
        strings = self.dataset.batch_to_strings(output)
        print('\nManual decoding results: ')
        for i, s in enumerate(strings):
            print('{} -> {}'.format(self.test_strings[i], s))
        print()


class RandomWindowBatchDecodingCallback(object):

    def __init__(self, iterator: PredictMiddleNoisedWindowIterator):
        self.iterator = iterator

    def __call__(self, model):
        random_batch = self.iterator.get_single_rnd_batch()
        output = model(random_batch.noised).softmax(dim=1).argmax(dim=1)
        buffer = ''.join(self.iterator.dataset.fields['text'].vocab.itos[c] for c in random_batch.clean.squeeze(1))
        print('Reference: {}'.format(buffer))
        buffer = ''.join(self.iterator.dataset.fields['text'].vocab.itos[c] for c in output)
        print('Result: {}'.format(buffer))
