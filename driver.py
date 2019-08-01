import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from cnn_ae.models.denoising import CNNEncoder, CNNDecoder, CNNAE, DownsamplingCNNEncoder, UpsamplingCNNDecoder
from cnn_ae.data.datasets import AutoencodingDataset
from cnn_ae.trainers.denoising import DenoisingCNN
from cnn_ae.trainers.callbacks import ManualTestingCallback
from torchtext.data.iterator import BucketIterator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='CNN character auto encoding')
parser.add_argument('--mode', action='store', choices=['train', 'debug'], required=True, type=str)
parser.add_argument('--dataset', action='store', type=str)
parser.add_argument('--topk', action='store', default=None, type=int)
parser.add_argument('--model', action='store', type=str)
parser.add_argument('--manual-examples', action='store', default=None, type=str)
parser.add_argument('--max-iter', action='store', default=1000, type=int)
params = parser.parse_args()

if params.mode == 'train':
    ds = AutoencodingDataset(params.dataset, params.topk)
    train, test = ds.split()
    if params.manual_examples:
        with open(params.manual_examples, 'r') as in_file:
            sentences = [line.strip() for line in in_file]
        callback = ManualTestingCallback(ds, sentences, device)
    else:
        callback = None

    train_iterator = BucketIterator(train, 64, sort_key=lambda x: len(x.text), device=device)
    test_iterator = BucketIterator(test, 64, sort_key=lambda x: len(x.text), device=device)

    enc = DownsamplingCNNEncoder(len(ds.fields['text'].vocab), 200)
    dec = UpsamplingCNNDecoder(len(ds.fields['text'].vocab), 200, 1024, 64, 3)
    model = CNNAE(enc, dec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = DenoisingCNN(model, optimizer, ds.fields['text'].vocab.stoi['<pad>'], train_iterator, test_iterator, params.max_iter, params.model, device)
    trainer.train(end_epoch_callback=callback)

