from utils import RatingsDataset, data_train_dir, data_test_dir
from model import BasicAutoencoder
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


def get_parser():
    """
    Creates an argument parser and assigns all the required user defined
    hyper parameters
    :return: parser
    """
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
    tied = args.tied
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
    dataset_train = RatingsDataset(add_noise, data_train_dir, batch_size=batch_size)
    dataset_test = RatingsDataset(add_noise, data_test_dir, batch_size=batch_size)
    data_loader = DataLoader(dataset_train, 1,
                             shuffle=True, num_workers=4,
                             pin_memory=True)
    model_sizes = [4000, 1000, 500, 100]
    model = BasicAutoencoder(tied_weights=tied, sizes=model_sizes,
                             activation=nn.ReLU, init_weights=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train loop
    for epoch in range(n_epochs):
        if epoch % verbosity == 0:
            print('Epoch', epoch, '/', n_epochs)

        for batch in data_loader:
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
