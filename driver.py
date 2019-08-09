import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from cnn_ae.models.denoising import CNNAE, ShallowDSCNNEncoder, ShallowUSCNNDecoder, RNNDecoder, CNNRNNAE
from cnn_ae.data.datasets import AutoencodingDataset
from cnn_ae.trainers.denoising import DenoisingCNN
from cnn_ae.trainers.callbacks import ManualTestingCallback
from torchtext.data.iterator import BucketIterator

# conv = nn.Conv1d(200, 1, 3)
# tconv = nn.ConvTranspose1d(1, 200, 3)
# tconv.weight = conv.weight
#
# output = conv(data)
# s = output.size()
# output, indices = F.max_pool1d(output, kernel_size=output.shape[-1], return_indices=True)
# output = F.max_unpool1d(output, indices, s[-1], output_size=s)
# output = tconv(output)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='CNN character auto encoding')
parser.add_argument('--mode', action='store', choices=['train', 'debug'], required=True, type=str)
parser.add_argument('--dataset', action='store', type=str)
parser.add_argument('--topk', action='store', default=None, type=int)
parser.add_argument('--model-best', action='store', type=str)
parser.add_argument('--model-end', action='store', default=None, type=str)
parser.add_argument('--load-from', action='store', choices=['none', 'best', 'end'], default='none', type=str)
parser.add_argument('--manual-examples', action='store', default=None, type=str)
parser.add_argument('--max-iter', action='store', default=1000, type=int)
parser.add_argument('--noise-ratio', action='store', default=0.0, type=float)
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

