

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


