import numpy as np
from clique_based import CliqueBased


class CARTBased(CliqueBased):
    def __init__(self):
        """
        Implements CART
        """
        super().__init__()

    def get_left_side_sum_squares(self, labels, local_counts, left_side):
        left_side.fill(0)
        for j in range(local_counts.size):
            cluster_counts = np.arange(1,local_counts[j]+1)
            left_side[labels == j] = cluster_counts**2 - np.pad(cluster_counts[:-1], (1,0))**2
        return left_side.cumsum()

    def get_left_side_wrt_node(self, labels, local_counts):
        left_side = np.zeros(labels.size)
        for j in range(local_counts.size):
            left_side[labels == j] = local_counts[j]
        return left_side.cumsum()

    def get_right_side_sum_squares(self, labels, local_counts):
        left_side_wrt_node = self.get_left_side_wrt_node(labels, local_counts)
        left_side_sum_squares = self.get_left_side_sum_squares(labels, local_counts, np.zeros(labels.size))
        total_sum_squares = (local_counts**2).sum()

        return total_sum_squares - 2*left_side_wrt_node + left_side_sum_squares   

    def get_metric(self, global_counts, local_counts, labels, coeff, left_side):
        size_v = local_counts.sum()
        size_left = np.arange(1, size_v+1)
        left_side_sum_squares = self.get_left_side_sum_squares(labels, local_counts, left_side)
        right_side_sum_squares = self.get_right_side_sum_squares(labels, local_counts)
        vol_left = (size_left**2 - left_side_sum_squares)
        vol_right = ((size_v - size_left)**2 - right_side_sum_squares)

        with np.errstate(divide='ignore', invalid='ignore'):
            res = ((vol_left/size_left) + (vol_right/(size_v - size_left))) / size_v
        return res

    def train_metric(self, cond, clustering, current_data, global_counts):
        local_counts = self.get_local_counts(clustering, current_data, global_counts.size)
        size_v = local_counts.sum()
        impurity_v = 1-((local_counts**2).sum()/(size_v**2))

        return cond - impurity_v
    