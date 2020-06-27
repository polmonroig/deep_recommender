from argparse import ArgumentParser
from model import BasicAutoencoder
from train import xavier_init
from os.path import join
import torch.nn as nn
import torch


def get_parser():
    """
    Creates an argument parser that defines two modes of execution:
        1. Evaluate a model by entering an input data in csv format
        2. Generate an ONNX model based on a provided model
    For the evaluation model a ONNX must be provided
    """
    parser = ArgumentParser()
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate a specified model')
    parser.add_argument('--generate', dest='evaluate', action='store_false', help='Convert a pytorch model to ONNX')
    parser.add_argument('--model', type=str, help='Model to convert or evaluate')
    parser.add_argument('--batch_size', type=int, help='Batch size of the input data')
    parser.add_argument('--data', type=str, help='Data to evaluate')
    return parser


def eval_model(model_path, data):
    raise NotImplementedError()

def convert_model(model_path, batch_size):
    model = BasicAutoencoder(tied_weights=True, sizes=[176275, 1000, 500, 100],
                             activation=nn.functional.relu,
                             w_init=xavier_init)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model_path.split('/'))
    input_names = [ 'input' ]
    output_names = [ 'output' ]
    model_path = model_path.split('/')
    model_name = model_path[-1]
    model_dir = ''
    for name in model_name:
        model_dir = join(model_dir, name)
    dummy_input = torch.randn(batch_size, 176275)
    torch.onnx.export(model, dummy_input, join(model_dir, model_name + '.onnx'),
                     export_params=True, verbose=True,
                     input_names=input_names, output_names=output_names)

def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate = args.evaluate
    model_path = args.model
    data = args.data
    batch_size = args.batch_size
    if evaluate:
        eval_model(model_path, data)
    else:
        convert_model(model_path, batch_size)



if __name__ == '__main__':
    main()
