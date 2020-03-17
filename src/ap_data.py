import csv
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


def load_edges(path):
    """
    Actually returns a similarity matrix
    :param path: path for Gowalla dataset with connections
    :return: sparse matrix that represent graph nodes and their connections
    """
    with open(path) as file:
        reader = csv.reader(file, delimiter="\t")
        raw_data = [(int(e), int(c)) for e, c in reader]  # e & c stands for edge and connection
        data_shape = (
        np.max([x for x in (itertools.chain(*raw_data))]) + 1, reader.line_num)  # num of edges, connections
        data_mat = sparse.coo_matrix(([1] * len(raw_data), list(zip(*raw_data))), shape=(data_shape[0], data_shape[0]))
    return data_mat.tocsr()


def load_checkins(path):
    checkins_df = pd.read_csv(path, sep="\t", header=None)[[0, 4]]
    checkins_df.columns = ["user_id", "location_id"]
    unique_users = np.unique(checkins_df.user_id)

    train_users, test_users = train_test_split(unique_users, test_size=0.01, shuffle=True)

    train = checkins_df.loc[checkins_df.user_id.isin(train_users)]
    test = checkins_df.loc[checkins_df.user_id.isin(test_users)]

    return train, test
