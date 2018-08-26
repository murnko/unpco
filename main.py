from src.utils import create_raw_data_df_list, create_csv_data
from src.customdataset import CustomDatasetFromImages
from src.utilities.cropper import crop_top
from src.autoencoder import VAE
import numpy as np
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.nn import functional as F
import argparse
import torch


parser = argparse.ArgumentParser(description='VAE Test')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                    help='how big is z')
parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                    help='how big is linear around z')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

path_bi_data = 'data/crop_data'
df_bi_labeled = create_csv_data(path_bi_data, [2])
print(df_bi_labeled.head())
msk = np.random.rand(len(df_bi_labeled)) < 0.8
train = df_bi_labeled[msk]
test = df_bi_labeled[~msk]
print(train['depth_2'].value_counts())
print(test['depth_2'].value_counts())

# print(df_bi_labeled.depth_2.value_counts())
ds_train = CustomDatasetFromImages(train)
ds_test = CustomDatasetFromImages(test)
# crop_top(df_bi_labeled)

train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                                batch_size=args.batch_size,
                                                shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                                batch_size=args.batch_size,
                                                shuffle=True)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if epoch == args.epochs and i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       'snapshots/conv_vae/reconstruction_' + str(epoch) +
                       '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


model = VAE(args)
if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch == args.epochs:
        sample = Variable(torch.randn(64, args.hidden_size))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 3, 32, 32),
                   'snapshots/conv_vae/sample_' + str(epoch) + '.png')

