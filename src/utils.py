from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# global data variables definition
data_dir = '../data'
data_files = sorted(os.listdir(data_dir))
data_credits = os.path.join(data_dir, 'credits.csv')
data_keywords = os.path.join(data_dir, 'keywords.csv')
data_links = os.path.join(data_dir, 'links.csv')
data_links_small = os.path.join(data_dir, 'links_small.csv')
data_movies_metadata = os.path.join(data_dir, 'movies_metadata.csv')
data_ratings = os.path.join(data_dir, 'ratings.csv')
data_ratings_small = os.path.join(data_dir, 'ratings_small.csv')
data_train = os.path.join(data_dir, 'train/data')
data_test = os.path.join(data_dir, 'test/data')

# global model definitions
model_dir = '../models'


class DataProcessor:
    """
    The dataset encapsulates all the data set and it is in charge
    of preforming any feature extraction, filtering and parsing
    needed for the training. It also separates the data into a
    train and test set that can be easily called. The complete dataset
    is not too big, but since we need to generate a sparse matrix with it
    the size grows inmensely. To solve this issue we separate the data into
    multiple files that must be loaded concurrently while training.

    Attributes
    self.test_split (float): the percentage of data that belongs to the test set
    self.seed (int): seed used to split the dataset
    self.users_per_file (int): max users per file of data
    self.transforms (array): list of transformations to apply to the data
    """
    def __init__(self, test_split=0.2, seed=42, users_per_file=128, transforms=None):
        self.test_split = test_split
        self.seed = seed
        self.users_per_file = 128
        self.transforms = transforms

    def prepare_and_write(self):
        """
        Reads the data from disks and performs
        all the data wrangling required
        :param transforms list of transformations to apply
        :return: None
        """
        # split dataset
        data = pd.read_csv(data_ratings)
        # split dataset
        train, test = train_test_split(data, test_size=self.test_split, random_state=self.seed)
        self.save_data(data_train, train)
        self.save_data(data_test, test)



    def save_data(self, name, data):
        users = data['userId']
        movies = data['movieId']
        ratings = data['rating']
        total_users = len(set(users.tolist()))
        total_movies = len(set(movies.tolist()))
        total_files = total_users // self.users_per_file
        user = 0
        entry = 0
        for file in range(total_files):
            users_in_file = min(self.users_per_file, self.users_per_file)
            max_users = users_in_file + user
            shape = (users_in_file, total_movies)
            dataset = np.zeros(shape, dtype=np.float32)
            while user < max_users:
                d = data.iloc[entry]
                entry_user = d['userId'] - 1
                while user == entry_user:
                    dataset[user][d['movieId'] - 1] = d['rating']
                    entry += 1
                    d = data.iloc[entry]
                    entry_user = d['userId'] - 1

                user += 1

            file_name = name + '_' + DataProcessor.padding(file, total_files) + '.csv'
            print('Saving file', file_name)
            np.savetxt(file_name, dataset, delimiter=',')

    @staticmethod
    def padding(number, size):
        file = str(number)
        size = str(size)
        while len(file) < len(size):
            file = '0' + file

        return file
