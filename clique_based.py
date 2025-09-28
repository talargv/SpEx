from base_classes import _Base_ExplainableTree, RefMixin, CutComp, Cut, GraphMixin
import numpy as np
import heapq

class CliqueBased(_Base_ExplainableTree, RefMixin, GraphMixin):
    def __init__(self):
        """
        Implements SpEx for clique graphs.
        The implementation takes advantage of the specific structure to avoid quadratic dependence on the number of samples.
        """
        super().__init__()

    def get_left_side(self, labels: np.ndarray, local_counts: np.ndarray, left_side: np.ndarray) -> np.ndarray:
        left_side.fill(0)
        for j in range(local_counts.size):
            left_side[labels == j] = np.arange(local_counts[j])
        return left_side.cumsum()
    
    def get_cum_counts(self, counts, labels):
        return counts[labels].cumsum()

    def get_deg(self, cum_counts, coeff):
        deg_left = cum_counts - coeff
        deg_right = deg_left[-1] - deg_left

        return deg_left, deg_right

    def train_metric(self, cond, clustering, current_data, global_counts):
        local_counts = self.get_local_counts(clustering, current_data, global_counts.size)
        ncut = (local_counts*(global_counts - local_counts)).sum() / (local_counts*(global_counts-1)).sum()
        return cond - ncut

    def get_metric(self, global_counts, local_counts, labels, coeff, left_side):
        cum_glob_counts = self.get_cum_counts(global_counts, labels)
        cum_loc_counts = self.get_cum_counts(local_counts, labels)
        left_side = self.get_left_side(labels, local_counts, left_side)

        deg_left, deg_right = self.get_deg(cum_glob_counts, coeff)
        cut_left = cum_glob_counts - 2*left_side - coeff
        cut_right = cut_left[-1] + 2*(cum_loc_counts - left_side) - cum_glob_counts - coeff

        with np.errstate(divide='ignore', invalid='ignore'):
            res = (cut_left / deg_left) + (cut_right / deg_right)
        return res

    def single_partition(self, data: np.ndarray, clustering: np.ndarray, current_data: np.ndarray, global_counts: np.ndarray) -> tuple[float, int, float]:
        best_cond, best_index, best_threshold = np.inf, 0, -np.inf 
        if current_data.size < 3:
            # There is a very gentle point where I assume there are at least samples
            # In any case splitting data of size < 3 seems irrelevant
            return best_cond, best_index, best_threshold

        local_counts = self.get_local_counts(clustering, current_data, global_counts.size)
        coeff = np.arange(1,current_data.size + 1)
        left_side = np.zeros(current_data.size)
        
        for i in range(data.shape[1]):
            data_ordering = self._order_current_data(data, current_data, i)
            labels = clustering[data_ordering]
            metric = self.get_metric(global_counts, local_counts, labels, coeff, left_side)

            metric[-1] = np.nan # I know this looks bad. This is to avoid bug caused by rounding error. 

            inner_data = data[data_ordering,i]
            best_data_point = self.best_data_point(metric, inner_data)
            current_best_cond = metric[best_data_point]

            if current_best_cond < best_cond:
                best_cond = current_best_cond
                best_threshold = inner_data[best_data_point]
                best_index = i
        return best_cond, best_index, best_threshold

    def train(self, data: np.ndarray, clustering: np.ndarray, k: int = None):
        """
        Trains a tree with given dataset and a reference clustering.
        Optionally, the number of leaves can be specified.

        Args:
            data: np.ndarray
                The full dataset.
                Dimensions should be (num samples, sample dimension).
            clustering: np.ndarray
                The clustering of the data.
                Dimensions should be (num samples,).
            k: int
                The number of leaves to train the tree with. If not specified, the number of clusters in the reference clustering is used.
        """
        _, clustering, global_counts = np.unique(clustering, return_inverse=True, return_counts=True)
        if k is None:
            k = global_counts.size

        current_data = np.arange(data.shape[0])
        heap = []

        clustering = clustering.astype(np.int64)
        current_data = current_data.astype(np.int64)
        
        best_cond, best_index, best_threshold = self.single_partition(data, clustering, current_data, global_counts)
        self.tree.root.coordinate = best_index
        self.tree.root.threshold = best_threshold
        heapq.heappush(heap, CutComp(best_cond, self.tree.root, current_data))

        for _ in range(k-1):
            node = heapq.heappop(heap)
            cut, current_data = node.cut, node.indices
            left_data, right_data = self._split(data, current_data, cut.coordinate, cut.threshold)

            l_best_cond, l_best_index, l_best_threshold = self.single_partition(data, clustering, left_data, global_counts)
            r_best_cond, r_best_index, r_best_threshold = self.single_partition(data, clustering, right_data, global_counts)

            cut_left = Cut(l_best_index, l_best_threshold)
            cut_right = Cut(r_best_index, r_best_threshold)

            cut.left = cut_left
            cut.right = cut_right

            l_train_metric = self.train_metric(l_best_cond, clustering, left_data, global_counts)
            r_train_metric = self.train_metric(r_best_cond, clustering, right_data, global_counts)

            heapq.heappush(heap, CutComp(l_train_metric, cut_left, left_data))
            heapq.heappush(heap, CutComp(r_train_metric, cut_right, right_data))
    