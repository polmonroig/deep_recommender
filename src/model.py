import torch.nn as nn


class BasicAutoencoder(nn.Module):
    """
    The basic autoencoder is neural network that can take different forms
    depending on the environment it works on. It is a feedforward deep
    neural network that is symmetrical in both ends and that it can have
    its encoder/decoder weights tied or not. If the weights are tied, the
    network will perform and train faster and the encoding operation
    will be the reverse of the decoding. On the other hand if the data
    that is trained on the network has a noise applied, the network becomes
    a denoising autoencoder.

    Attributes:
        self.tied_weights (bool): indicates that the encoder and decoder weights are tied
        self.encoder_layers (list): contains a list of linear layers and
                                    activation functions for the encoder
        self.decoder_layers (list): contains a list of linear layers and
                                    activation functions for the encoder
        self.sizes (list): contains a list of sizes that represent the number of neurons
                           for each layer; given an index 'i' of the list of sizes
                           the layer i will contain self.sizes[i] input features and
                           self.sizes[i + 1] output features
    """
    def __init__(self, tied_weights, sizes, activation, init_weights):
        super().__init__()
        self.tied_weights = tied_weights
        self.sizes = sizes
        self.activation = activation
        self.encoder_layers = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.encoder_layers.append(nn.Linear(in_features=sizes[i - 1], out_features=sizes[i]))
        self.decoder_layers = nn.ModuleList()
        for i in range(len(sizes) - 1, 0, -1):
            self.decoder_layers.append(nn.Linear(in_features=sizes[i], out_features=sizes[i - 1]))
        if self.tied_weights:
            for encoder_layer, decoder_layer in zip(self.encoder_layers, reversed(self.decoder_layers)):
                decoder_layer.weight.data = encoder_layer.weight.data.transpose(0,1)



    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        return x

    def decode(self, x):
        for layer in self.decoder_layers:
            x = self.activation(layer(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
