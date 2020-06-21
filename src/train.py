from utils import RatingsDataset
from argparse import ArgumentParser
import torch


def main():
    print("Training initialized")
    # hyper parameter parsing
    parser = ArgumentParser()
    parser.add_argument('--denoiser', dest='feature', action='store_true', help='Use denoiser autoencoder')
    parser.add_argument('--basic', dest='feature', action='store_false', help='Use basic autoencoder')

    args = parser.parse_args()
    add_noise = args.denoise
    # device initialization
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on gpu')
    else:
        device = torch.device('cpu')
        print('Running on cpu')

    dataset = RatingsDataset(add_noise)


if __name__ == '__main__':
    main()
