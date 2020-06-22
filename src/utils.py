from sklearn.model_selection import train_test_split
from scipy import sparse
from torch.utils.data import Dataset
from random import randint
import torch
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
        self.users_per_file = users_per_file
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
        self.save_data(data_train, train, data['movieId'])
        self.save_data(data_test, test, data['movieId'])

    def save_data(self, name, data, movies):
        """
        The training/testing data is represented with a sparse
        matrix where each row is a user and each col an item (i.e. movie)
        The complete sparse matrix is extremely big, even though it is almost
        empty, it has a lot of zeros, to prevent a data overflow we must divide
        the data into different chunks, each chunk will contain a certain users,
        for purpose of this project we don't really care if user A is in row 1
        or user B in col 2, we only care about its features. In other words
        we must save just the content of each user in each row, independently of
        which user we are saving.

        A second improvement was needed to save space was to compress each file
        as a special format 'npz' for sparse matrixes.

        A problem I encountered is that each movie has an id assigned but the set
        that contains the ids of movies watched by the user does not contain all the movies,
        that is why the total movies is calculated as the max possible value of the id.
        On the other hand, we also pass the data of the movies of the entire dataset.

        """
        users = data['userId']
        ratings = data['rating']
        total_users = len(set(users.tolist()))
        total_movies = np.array(movies.tolist()).max()

        total_files = total_users // self.users_per_file
        print('Generating a total of', total_files, 'files')
        user = 0
        entry = 0
        for file in range(total_files):
            users_in_file = min(self.users_per_file, total_users - user)
            current_user = 0
            max_users = users_in_file + user
            shape = (users_in_file, total_movies)
            dataset = np.zeros(shape, dtype=np.float32)
            while current_user < users_in_file:
                d = data.iloc[entry]
                entry_user = d['userId'] - 1
                while user == entry_user:
                    dataset[current_user][int(d['movieId'] - 1)] = d['rating']
                    entry += 1

                    d = data.iloc[entry]
                    entry_user = d['userId'] - 1

                current_user += 1
                user += 1

            file_name = name + '_' + DataProcessor.padding(file, total_files)
            print('Saving file', file_name)
            sparse_dataset = sparse.csc_matrix(dataset)
            sparse.save_npz(file_name, sparse_dataset)

    @staticmethod
    def padding(number, size):
        file = str(number)
        size = str(size)
        while len(file) < len(size):
            file = '0' + file

        return file


class RatingsDataset(Dataset):
    """
    The ratings dataset contains all the different train files, each file contains
    a quantity of N users (except the last file which might contain less).

    For a given list of indices we need to calculate which users to return. Given that
    we only have a list of files, we must specify that the batch size, this size will be
    the number of users per file, thus each index represents a file, the batch size of
    the data loader will specify the number of files. As a precondition, the batch size
    is always lower than the number of users in a file.
    """

    def __init__(self, add_noise, directory, batch_size):
        self.add_noise = add_noise
        self.batch_size = batch_size
        self.files = [os.path.join(directory, file) for file in os.listdir(directory)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = sparse.load_npz(self.files[item]).todense()
        data = torch.from_numpy(data)
        length = data.shape[0]
        max_perm = randint(self.batch_size, length)
        indices = torch.randperm(self.batch_size)
        indices += max_perm

        return data[indices]
