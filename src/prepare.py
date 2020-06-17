import utils


def main():
    print('Preparing data...')
    transforms = []
    data = utils.Data(transforms=transforms)
    data.prepare_and_write()


if __name__ == '__main__':
    main()
