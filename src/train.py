from utils import RatingsDataset
from model import BasicAutoencoder
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import torch.optim as optim


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--denoiser', dest='autoencoder', action='store_true', help='Use denoiser autoencoder')
    parser.add_argument('--basic', dest='autoencoder', action='store_false', help='Use basic autoencoder')
    parser.add_argument('--tied_weights', dest='tied', action='store_true', help='Tie encoder/decoder weights')
    parser.add_argument('--not_tied', dest='tied', action='store_false', help='Use separate encoder/decoder weights')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs for the training loop')
    parser.add_argument('--batch_size', type=int, help='Batch size of the training and eval set')
    parser.add_argument('--verbose_epochs', type=int, help='Number of epochs per training verbose output')
    parser.add_argument('--lr', type=float, help='Learning rate of the optimizer')
    return parser


def eval_step(model, x, y):
    raise NotImplementedError('Eval step currently not implemented')


def train_step(model, optimizer, x, y):
    raise NotImplementedError('Train step currently not implemented')


def main():
    print("Training initialized")
    # hyper parameter parsing
    parser = get_parser()
    args = parser.parse_args()
    add_noise = args.autoencoder
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    verbosity = args.verbose_epochs
    learning_rate = args.lr

    # device initialization
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on gpu')
    else:
        device = torch.device('cpu')
        print('Running on cpu')
    # dataset creation
    dataset = RatingsDataset(add_noise)
    dataloader = DataLoader(dataset, batch_size,shuffle=False, num_workers=4)
    model = BasicAutoencoder().to(device)
    optimizer = optim.Adam(model.params(), lr=learning_rate)

    # train loop
    for epoch in range(n_epochs):
        if epoch % verbosity == 0:
            print('Epoch', epoch, '/', n_epochs)

        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            model.train()
            train_step(model, optimizer, x, y)
            model.eval()
            eval_step(model, x, y)
            error = 0
        if epoch % verbosity == 0:
            print('Error', error)


if __name__ == '__main__':
    main()
