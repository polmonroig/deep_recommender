import utils


def main():
    print('Preparing data...')
    transforms = []
    data = utils.DataProcessor(transforms=transforms, users_per_file=500)
    data.prepare_and_write()


if __name__ == '__main__':
    main()
