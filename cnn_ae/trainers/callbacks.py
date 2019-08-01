class ManualTestingCallback(object):

    def __init__(self, dataset, test_strings):
        self.dataset = dataset
        self.test_strings = test_strings
        self.test_tensor = dataset.manual_strings_to_batch(*test_strings).permute(1, 0)

    def __call__(self, model):
        output = model(self.test_tensor).softmax(dim=2).argmax(dim=2).permute(1, 0)
        strings = self.dataset.batch_to_strings(output)
        print('Manual decoding results: ')
        for i, s in enumerate(strings):
            print('{} -> {}'.format(self.test_strings[i], s))
