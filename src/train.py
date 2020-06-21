from utils import RatingsDataset
from argparse import ArgumentParser
import torch


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--denoiser', dest='autoencoder', action='store_true', help='Use denoiser autoencoder')
    parser.add_argument('--basic', dest='autoencoder', action='store_false', help='Use basic autoencoder')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs for the training loop')
    parser.add_argument('--batch_size', type=int, help='Batch size of the training and eval set')
    parser.add_argument('--verbose_epochs', type=int, help='Number of epochs per training verbose output')
    return parser


def eval_step(model):
    raise NotImplementedError('Eval step currently not implemented')


def train_step(model):
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

    # train loop
    for epoch in range(n_epochs):
        if epoch % verbosity == 0:
            print('Epoch', epoch, '/', n_epochs)
        
        error = 0
        if epoch % verbosity == 0:
            print('Error', error)

if __name__ == '__main__':
    main()
