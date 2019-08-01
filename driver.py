import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cnn_ae.models.denoising import CNNEncoder, CNNDecoder, CNNAE, DownsamplingCNNEncoder, UpsamplingCNNDecoder
from cnn_ae.data.datasets import AutoencodingDataset
from cnn_ae.trainers.denoising import DenoisingCNN
from cnn_ae.trainers.callbacks import ManualTestingCallback
from torchtext.data.iterator import BucketIterator




device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ds = AutoencodingDataset('temp/dataset/BookCorpus_unique_char_256.txt', 10000)
train, test = ds.split()
callback = ManualTestingCallback(ds, ["this is a sentence", "i hope this works"])
# test = ds.manual_strings_to_batch("allo haha", "dudes")
# s = ds.batch_to_strings(test)

train_iterator = BucketIterator(train, 64, sort_key=lambda x: len(x.text), device=device)
test_iterator = BucketIterator(test, 64, sort_key=lambda x: len(x.text), device=device)

enc = DownsamplingCNNEncoder(len(ds.fields['text'].vocab), 200)
dec = UpsamplingCNNDecoder(len(ds.fields['text'].vocab), 200, 1024, 64, 3, 64)
model = CNNAE(enc, dec).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = DenoisingCNN(model, optimizer, ds.fields['text'].vocab.stoi['<pad>'], train_iterator, test_iterator, 1000, 'temp/saved_models/best.model', device)
trainer.train(end_epoch_callback=callback)

# data_batch = torch.randint(0, 10, (64, 256))
#
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
#
# for i in range(1000):
#     optimizer.zero_grad()
#     output = model(data_batch)
#     loss = criterion(output.view(-1, 10), data_batch.view(-1))
#     loss.backward()
#     optimizer.step()
#     print(loss.item())
#
# i = 0

