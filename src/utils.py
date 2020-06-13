import os

# global variables definition
data_dir = '../data'
data_files = sorted(os.listdir(data_dir))
data_credits = os.path.join(data_dir, 'credits.csv')
data_keywords = os.path.join(data_dir, 'keywords.csv')
data_links = os.path.join(data_dir, 'links.csv')
data_links_small = os.path.join(data_dir, 'links_small.csv')
data_movies_metadata = os.path.join(data_dir, 'movies_metadata.csv')
data_ratings = os.path.join(data_dir, 'ratings.csv')
data_ratings_small = os.path.join(data_dir, 'ratings_small.csv')


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
    def __init__(self, test_split):
        self.test_split = test_split
        self.train = None
        self.test = None

    def read(self):
        """
        Reads the data from disks and performs
        all the data wranling required
        :return: None
        """
        self.train = None
        self.test = None

    def train(self):
        return self.train

    def test(self):
        return self.test

