import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from cnn_ae.models.denoising import CNNAE, ShallowDSCNNEncoder, ShallowUSCNNDecoder, RNNDecoder, CNNRNNAE
from cnn_ae.models.window import MLP, CNN, ResMLP
from cnn_ae.trainers.window import WindowCorrectionTrainer
from cnn_ae.data.datasets import AutoencodingDataset, SplittableLanguageModelingDataset, RandomizedTextWindowDataset
from cnn_ae.trainers.denoising import DenoisingCNN
from cnn_ae.trainers.callbacks import ManualTestingCallback, RandomWindowBatchDecodingCallback
from torchtext.data.iterator import BucketIterator
from torchtext.data.field import Field
from cnn_ae.data.iterators import PredictMiddleNoisedWindowIterator, NoisedPreWindowedIterator
from python_utilities.utils.utils_fn import print_kv_box
from cnn_ae.utils.tokenize import WordToCharTokenizer
from cnn_ae.common.enums import Regularization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='CNN character auto encoding')
parser.add_argument('--mode', action='store', choices=['train', 'train_predict', 'debug'], required=True, type=str)
parser.add_argument('--dataset', action='store', type=str)
parser.add_argument('--topk', action='store', default=float('inf'), type=int)
parser.add_argument('--model-best', action='store', type=str)
parser.add_argument('--model-end', action='store', default=None, type=str)
parser.add_argument('--load-from', action='store', choices=['none', 'best', 'end'], default='none', type=str)
parser.add_argument('--manual-examples', action='store', default=None, type=str)
parser.add_argument('--max-iter', action='store', default=1000, type=int)
parser.add_argument('--noise-ratio', action='store', default=0.0, type=float)
parser.add_argument('--batch-size', action='store', default=128, type=int)
parser.add_argument('--window-size', action='store', default=51, type=int)
parser.add_argument('--hidden-size', action='store', default=512, type=int)
parser.add_argument('--depth', action='store', default=1, type=int)
parser.add_argument('--regularization', action='store', choices=['DROPOUT', 'BN_RELU', 'RELU_BN'], default='DROPOUT',
                    type=str)
parser.add_argument('--dropout', action='store', default=0.5, type=float)
parser.add_argument('--wd', action='store', default=1e-4, type=float)
parser.add_argument('--model-conf', action='store', default=None, type=str)
params = parser.parse_args()

kvs = [(k, v) for k, v in vars(params).items()]
kvs.append(('Device', device))

print_kv_box('Current Configuration', kvs)


if params.mode == 'debug':
    tokenizer = WordToCharTokenizer()
    text_field = Field(tokenize=tokenizer, batch_first=True)
    ds = RandomizedTextWindowDataset(params.dataset, text_field, params.window_size, topk=params.topk, newline_eos=False)
    text_field.build_vocab(ds)
    train_ds, test_ds = ds.split(0.8)
    iterator = NoisedPreWindowedIterator(train_ds, params.batch_size, params.window_size, 0.0)
    iterator = PredictMiddleNoisedWindowIterator(iterator, 1)
    for b in iterator:
        print(b)
    i = 1
    # model = MLP(51, 27, 1024, 3)
    # text_field = Field(tokenize=tokenize, batch_first=True)
    # ds = SplittableLanguageModelingDataset(params.dataset, text_field, newline_eos=False)
    # text_field.build_vocab(ds)
    # train, test = ds.split()
    # model = MLP(51, len(text_field.vocab), 1024, 3)
    # iterator = PredictMiddleNoisedWindowIterator(ds, 64, 51, 0.1, 1)
    # for b in iterator:
    #     output = model(b.noised, b.clean)

elif params.mode == 'train_predict':
    tokenizer = WordToCharTokenizer()
    text_field = Field(tokenize=tokenizer, batch_first=True)
    # ds = SplittableLanguageModelingDataset(params.dataset, text_field, topk=params.topk, newline_eos=False)
    ds = RandomizedTextWindowDataset(params.dataset, text_field, params.window_size, topk=params.topk, newline_eos=False)
    text_field.build_vocab(ds)
    train_ds, test_ds = ds.split(0.8)

    batch_size = params.batch_size
    window_size = params.window_size
    middle_width = 1

    train_iterator = NoisedPreWindowedIterator(train_ds, batch_size, window_size, params.noise_ratio, device=device)
    train_iterator = PredictMiddleNoisedWindowIterator(train_iterator, middle_width)
    test_iterator = NoisedPreWindowedIterator(test_ds, batch_size, window_size, params.noise_ratio, device=device, shuffle=False)
    test_iterator = PredictMiddleNoisedWindowIterator(test_iterator, middle_width)

    callback = None #RandomWindowBatchDecodingCallback(test_iterator)

    if params.model_conf:
        # model = CNN.from_conf(params.model_conf, window_size, len(text_field.vocab)).to(device)
        model = ResMLP(params.model_conf, window_size, len(text_field.vocab)).to(device)
    else:
        model = CNN(window_size, len(text_field.vocab), 2).to(device)
    # model = MLP(window_size, len(text_field.vocab), params.hidden_size, params.depth, dropout=params.dropout,
    #             regularization=Regularization[params.regularization]).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=params.wd)
    if params.load_from == 'best':
        model.load_state_dict(torch.load(params.model_best))
    elif params.load_from == 'end':
        model.load_state_dict(torch.load(params.model_end))

    kvs = [
        ('Dataset Length', len(ds.examples[0].text)),
        ('Batch Size', batch_size),
        ('Window Size', window_size),
        ('Num. Train Batches', len(train_iterator)),
        ('Num. Test Batches', len(test_iterator)),
    ]

    print_kv_box('Pre-training stats', kvs)

    trainer = WindowCorrectionTrainer(model, optimizer, train_iterator, test_iterator, params.max_iter,
                                      params.model_best, params.model_end, device)
    trainer.train(end_epoch_callback=callback)


elif params.mode == 'train':
    ds = AutoencodingDataset(params.dataset, params.topk, add_init_eos=False)
    train, test = ds.split()
    if params.manual_examples:
        with open(params.manual_examples, 'r') as in_file:
            sentences = [line.strip() for line in in_file]
        callback = ManualTestingCallback(ds, sentences, device)
    else:
        callback = None

    train_iterator = BucketIterator(train, 128, sort_key=lambda x: len(x.text), device=device)
    test_iterator = BucketIterator(test, 128, sort_key=lambda x: len(x.text), device=device)

    # kcn = [
    #     (5, 400, 3),
    #     (5, 400, 3),
    #     (5, 400, 3),
    #     (5, 400, 3)
    # ]
    # pooling_ks = [
    #     (2, 2),
    #     (2, 2),
    #     (2, 2),
    #     (-1, 1)
    # ]

    kernel_channels = [
        (3, 200),
        (4, 200),
        (5, 200),
        (6, 200),
        (7, 200)
    ]

    enc = ShallowDSCNNEncoder(len(ds.fields['text'].vocab), 200, kernel_channels, 2)
    dec = ShallowUSCNNDecoder(len(ds.fields['text'].vocab), 200, 2048, kernel_channels, device, 2)
    dec.tie_weights(enc.get_cnn_weights())
    model = CNNAE(enc, dec).to(device)
    # enc = ShallowDSCNNEncoder(len(ds.fields['text'].vocab), 200, kernel_channels)
    # dec = RNNDecoder(len(ds.fields['text'].vocab), 200, 1024, 200, 1024, device)
    # dec.tie_embedding(enc.emb.weight)
    # model = CNNRNNAE(enc, dec, ds.fields['text'].vocab.stoi['<START>'], 0.3, device).to(device)
    if params.load_from == 'best':
        model.load_state_dict(torch.load(params.model_best))
    elif params.load_from == 'end':
        model.load_state_dict(torch.load(params.model_end))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #ds.fields['text'].vocab.stoi['<pad>']
    trainer = DenoisingCNN(model, optimizer, train_iterator, test_iterator, params.max_iter, params.model_best, params.model_end, device)
    trainer.train(noise_ratio=params.noise_ratio, end_epoch_callback=callback)

