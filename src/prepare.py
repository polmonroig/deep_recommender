import utils 
import os 

def main():
    print('Preparing data...')
    data = utils.Data() 
    transforms = []
    print('Reading data...')
    data.read(transforms)
    # saves data as train/test .csv in data directory 
    print('Saving data...')
    data.write() 




if __name__ == '__main__':
    main()
