from utils import RatingsDataset, data_train_dir, data_test_dir
from model import BasicAutoencoder
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential
import torch
import wandb
import os
import gc
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


def loss_function(y_pred, y_true):
    loss = nn.functional.mse_loss
    indices = y_true != 0

    return loss(y_pred[indices], y_true[indices])

def eval_step(model, data_loader, device, verbosity):
    model.eval()
    criterion = loss_function
    for i, batch in enumerate(data_loader):
        data, target = batch
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data)
        if i % verbosity == 0:
            print('[' + str(i) + '/' + str(len(data_loader)) + ']')
            print('Test Loss', loss.item())
            wandb.log({"EvalLoss": loss})
        # memory management
        del loss, out, data, target
        gc.collect()
        torch.cuda.empty_cache()


def train_step(model, data_loader, optimizer, device, verbosity):
    model.train()
    criterion = loss_function
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        data, target = batch
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data)
        loss.backward()
        if i % verbosity == 0:
            print('[' + str(i) + '/' + str(len(data_loader)) + ']')
            print('Train Loss', loss.item())
            wandb.log({"TrainLoss": loss})
        # memory management
        del loss, out, data, target
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.step()


def xavier_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


def main():
    print('Training initialized')
    wandb.init(project="mrs")
    # hyper parameter parsing
    parser = get_parser()
    args = parser.parse_args()
    tied = args.tied
    add_noise = args.autoencoder
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    verbosity = args.verbose_epochs
    learning_rate = args.lr
    # log hyperparameters

    wandb.config.n_epochs = n_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.denoised = add_noise
    wandb.config.tied = tied

    # device initialization
    device = None
    if torch.cuda.is_available():
        device = torch.device('cpu')
        print('Running on cpu')
    else:
        device = torch.device('cpu')
        print('Running on cpu')
    # dataset creation
    dataset_train = RatingsDataset(add_noise, data_train_dir, batch_size=batch_size)
    dataset_test = RatingsDataset(add_noise, data_test_dir, batch_size=batch_size)
    train_data_loader = DataLoader(dataset_train, 1,
                             shuffle=True, num_workers=4,
                             pin_memory=True)
    test_data_loader = DataLoader(dataset_test, 1,
                             shuffle=True, num_workers=4,
                             pin_memory=True)
    model_sizes = [176275, 1000, 500, 100]
    model = BasicAutoencoder(tied_weights=tied, sizes=model_sizes,
                             activation=nn.functional.relu, w_init=xavier_init).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # train loop
    for epoch in range(n_epochs):
        print('Epoch', epoch, '/', n_epochs)
        train_step(model, train_data_loader, optimizer, device, verbosity)
        eval_step(model, test_data_loader, device, verbosity)
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        scheduler.step()


if __name__ == '__main__':
    main()
