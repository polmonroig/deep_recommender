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
data_train = os.path.join(data_dir, 'train.csv')
data_test = os.path.join(data_dir, 'test.csv')

# global model definitions
model_dir = '../models'


class Data:
    """
    The dataset encapsulates all the data set and it is in charge
    of preforming any feature extraction, filtering and parsing
    needed for the training. It also separates the data into a
    train and test set that can be easily called. Since the data
    is small enough, it can be read completely before training,
    it does not involve any parallel read/train architecture
    such as with torch.utils.data.

    Attributes
    self.test_split (float): the percentage of data that belongs to the test set
    self.train (np.array): contains the training set
    self.test (np.array): contains the test/eval set
    """
    def __init__(self, test_split=0.2, seed=42):
        self.test_split = test_split
        self.train = None
        self.test = None
        self.seed = seed

    def read(self):
        """
        Reads the already prepared data from disk,
        call this function when training the network
        """
        self.train = pd.read_csv(data_train)
        self.test = pd.read_csv(data_test)

    def read_and_prepare(self, transforms):
        """
        Reads the data from disks and performs
        all the data wrangling required
        :param transforms list of transformations to apply
        :return: None
        """
        features, target = Data.read_unprocessed()
        # apply transforms
        for t in transforms:
            features = t(features)
        # split dataset
        result = train_test_split(features, target,
                                  test_size=self.test_split, random_state=self.seed)

        self.train = (result[0], result[2])
        self.test = (result[1], result[3])

    @staticmethod
    def read_unprocessed():
        """
        Read features and target for processing, we are working in
        an unsupervised task with collaborative filtering so
        the features must be equal to the target
        :return:
        """
        data = pd.read_csv(data_ratings_small)
        users = data['userId']
        movies = data['movieId']
        ratings = data['rating']
        total_movies = len(set(movies.tolist()))
        total_users = len(set(users.tolist()))
        print('Features size:', (total_users, total_movies))
        features = np.zeros((total_users, total_movies), dtype=np.float32)
        for user, rating in zip(users, ratings):
            features[user - 1] = rating
        target = features

        return features, target

    def write(self):
        """
        Saves the processed train  and test set to the data directory
        :return: None
        """
        np.savetxt(data_train, self.train[0], delimiter=',')
        np.savetxt(data_test, self.test[1], delimiter=',')

    def train(self):
        return self.train

    def test(self):
        return self.test
