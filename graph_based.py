from base_classes import _Base_ExplainableTree, GraphMixin, CutComp, Cut

import numpy as np
import heapq
import scipy.sparse as sprs

class GraphBased(_Base_ExplainableTree, GraphMixin):
    """
        Implements SpEx for general graphs.
        To avoid long running times and memory usage, the implementation assumes that the graph is sparse.
    """
    def __init__(self):
        super().__init__()

    def get_metric(self, graph, data_ordering, total_deg_vector):
        total_deg_left = total_deg_vector[data_ordering].cumsum()
        total_deg = total_deg_left[-1]
        inner_right_neighbors, inner_left_neighbors, total_inner_deg = self.get_inner_neighbors(graph, data_ordering)

        with np.errstate(divide='ignore', invalid='ignore'):
            res = (2*inner_right_neighbors - total_inner_deg) / (total_deg - total_deg_left) - 2*(inner_left_neighbors / total_deg_left)
        return res
    
    def train_metric(self, cond, graph, current_data):
        ncut = 1 - (graph[np.ix_(current_data, current_data)].sum() / graph[current_data, :].sum())
        return cond - ncut

    def get_inner_neighbors(self, graph: sprs.csr_matrix, data_ordering: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        inner_graph = graph[np.ix_(data_ordering, data_ordering)]
        upper = sprs.triu(inner_graph)
        lower = upper.T
        inner_right_neighbors = upper.sum(axis=1).cumsum().A1
        inner_left_neighbors = lower.sum(axis=1).cumsum().A1

        return inner_right_neighbors, inner_left_neighbors ,inner_right_neighbors[-1] + inner_left_neighbors[-1]

    def single_partition(self, data: np.ndarray, graph: sprs.csr_matrix, current_data: np.ndarray, total_deg_vector: np.ndarray) -> tuple[float, int, float]:
        best_cond, best_index, best_threshold = np.inf, 0, -np.inf 
        if current_data.size < 3:
            # There is a very gentle point where I assume there are at least samples
            # In any case splitting data of size < 3 seems irrelevant
            return best_cond, best_index, best_threshold
        

        for i in range(data.shape[1]):
            data_ordering = self._order_current_data(data, current_data, i)     

            metric = self.get_metric(graph, data_ordering, total_deg_vector)
            inner_data = data[data_ordering,i]

            best_data_point = self.best_data_point(metric, inner_data)
            current_best_cond = metric[best_data_point]
            
            if current_best_cond < best_cond:
                best_cond = current_best_cond
                best_threshold = inner_data[best_data_point]
                best_index = i

        return best_cond, best_index, best_threshold

    def train(self, data: np.ndarray, graph: sprs.csr_matrix, k: int):
        """
        Trains a tree with k leaves on a given dataset and a reference graph.

        Args:
            data: np.ndarray
                The full dataset.
                Dimensions should be (num samples, sample dimension).
            graph: sprs.csr_matrix
                The full adjacency matrix of the graph to split on.
                The nodes correspond to the samples in the data.
            k: int
                The number of leaves to train the tree with.
        """
        assert sprs.issparse(graph) # Can still work with np arrays, with a few changes
        heap = []

        current_data = np.arange(data.shape[0])
        deg_vector = graph.sum(axis=1).A1
        
        best_cond, best_index, best_threshold = self.single_partition(data, graph, current_data, deg_vector)
        self.tree.root.coordinate = best_index
        self.tree.root.threshold = best_threshold
        heapq.heappush(heap, CutComp(best_cond, self.tree.root, current_data))

        for _ in range(k-1):
            node = heapq.heappop(heap)
            cut, current_data = node.cut, node.indices
            left_data, right_data = self._split(data, current_data, cut.coordinate, cut.threshold)

            l_best_cond, l_best_index, l_best_threshold = self.single_partition(data, graph, left_data, deg_vector)
            r_best_cond, r_best_index, r_best_threshold = self.single_partition(data, graph, right_data, deg_vector)

            cut_left = Cut(l_best_index, l_best_threshold)
            cut_right = Cut(r_best_index, r_best_threshold)

            cut.left = cut_left
            cut.right = cut_right

            l_train_metric = self.train_metric(l_best_cond, graph, left_data)
            r_train_metric = self.train_metric(r_best_cond, graph, right_data)

            heapq.heappush(heap, CutComp(l_train_metric, cut_left, left_data))
            heapq.heappush(heap, CutComp(r_train_metric, cut_right, right_data))
    