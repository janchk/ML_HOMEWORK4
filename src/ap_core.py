from scipy import sparse
from tqdm import tqdm
import numpy as np
import pandas as pd


class Ap:
    def __init__(self):
        self.M = None
        self.s = None
        self.a = None
        self.r = None
        self.labels = None
        self.shape = None
        self.clust_loc = None
        self.iters = None

    def fit(self, M: sparse.csr_matrix, iters_num):
        self.iters = iters_num
        self.s = M
        self.shape = M.shape

        self.a = sparse.csr_matrix(self.shape)
        """We use list of lists initialization for self.r coz we change 
        value of responsibility matrix thus we make a change to sparse structure
        which is expensive operation to do with csr matrix"""
        self.r = sparse.lil_matrix(self.shape)

        for i in tqdm(range(self.iters)):
            self.__update_responsibility()
            self.__update_availability()
        self.__get_clusters()

    def predict(self, checkins: pd.DataFrame, user_id=None, n=10):
        if self.clust_loc is None:
            checkins["cluster"] = self.labels[checkins.user_id]
            self.clust_loc = checkins.groupby(by="cluster")["location_id"].value_counts()

        if user_id:
            target_cluster = self.labels[user_id]
            try:
                topn_locs_clusterwise = list(self.clust_loc[target_cluster][:n].index)
            except IndexError:
                return list(checkins["location_id"].value_counts().index[:n])
            topn_locs = topn_locs_clusterwise
        else:
            topn_locs = list(checkins["location_id"].value_counts().index[:n])

        return topn_locs

    def __get_clusters(self):
        instances = (np.ravel(self.a.diagonal()) + np.ravel(self.r.diagonal())) > 0
        exemplars_idx = np.flatnonzero(instances)
        n_clusters = len(exemplars_idx)
        labels = np.ravel(self.s[:, exemplars_idx].argmax(-1))
        labels[exemplars_idx] = np.arange(n_clusters)  # getting cluster_num by  user_num
        clusters = [np.where(labels == i)[0] for i in range(n_clusters)]
        self.labels = labels
        self.clusters = clusters
        return clusters, instances

    def _get_similarity(self):
        """
        Getting Similarity matrix
        :return:
        """

        pass

    def __update_availability(self):
        self_resp = np.ravel(self.r.diagonal())
        _a_temp = sparse.lil_matrix(self.r.shape)
        _a = None
        _r = self.r.copy()
        a_diag = np.zeros(self.r.shape[0])

        _r_diag = np.ravel(_r.diagonal())
        _r.setdiag(0)
        _r_pos = _r.maximum(0).tocoo()  # getting positive mat
        _r_sum = np.ravel(_r_pos.sum(0))  # sum over columns

        for row, col, data in zip(_r_pos.row, _r_pos.col, _r_pos.data):
            if col != row:
                a_diag[col] += data
            _a = _r_diag[col] + _r_sum[col] - data
            # if _a > 0:
            #     print(_a)
            _a_temp[row, col] = _a if _a < 0 else 0

        self.a = _a_temp.minimum(0).tocoo()
        self.a.setdiag(a_diag)

    def __update_responsibility(self):
        """
        Filling "self.r value"
        :return:
        """
        _m_as = self.a + self.s  # Sum availability and similarity
        _m_as_2 = _m_as.copy()  # matrix a + s (aux value)

        enum = np.arange(self.shape[0])
        max_ind = np.ravel(np.argmax(_m_as, -1))

        _m_as_2[enum, max_ind] = -np.inf

        s_max = np.ravel(_m_as[enum, max_ind])
        rem_s_max = np.ravel(_m_as_2.max(-1).todense())
        # rem_s_max = np.max(np.ravel(_m_as_2)) # remaining max

        _m_as = _m_as.tocoo() # we need this transformation to be able to iterate over rows and columns

        for row, col, data in zip(_m_as.row, _m_as.col, _m_as.data):
            self.r[row, col] = data - rem_s_max[row] if data == s_max[row] else data - s_max[row]
