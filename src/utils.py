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



def get_data(test_split=0.2):
    """
    Generates a data quadruple that contains the data
    separated in a train/test split and it is formatted
    to
    :param test_split:
    :return:
    """
    raise NotImplementedError()
