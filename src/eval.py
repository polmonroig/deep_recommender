from argparse import ArgumentParser

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
    parser.add_argument('--data', type=str, help='Data to evaluate')


def eval_model(model, data):
    raise NotImplementedError()

def convert_model(model):
    raise NotImplementedError()

def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate = args.evaluate
    model = args.model
    data = args.data
    if evaluate:
        eval_model(model, data)
    else:
        convert_model(model)



if __name__ == '__main__':
    main()
