from numba import njit, prange, int32, int64
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class Cut():
    def __init__(self, coordinate: int = 0, threshold: float = np.inf, left = None, right = None, cluster = -1):
        """
        A tree node. Each such node is associated with an axis-aligned threshold cut. 

        In the case where the node is a leaf (left == right == None), coordinate and threshold might refer to the most promising split,
        determined by the training algorithm.
        """
        self.coordinate = coordinate
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cluster = cluster

    def __repr__(self):
        return f"({self.coordinate}, {self.threshold:.2f})"

class Tree():
    def __init__(self):
        self.root = Cut()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Clusters input data using the tree structure.
        Clusters are formed by passing the data through the tree, labeling samples that reached the same leaf with a unique label.

        Args:
            data: np.ndarray
                The data to cluster.
                Dimensions should be (num samples, sample dimension).

        Raises ValueError: If the input data's dimension are smaller than one of the split coordinates a tree node holds.
        """
        current_data = np.arange(data.shape[0])
        queue = [(self.root, current_data)]
        clustering = np.zeros(data.shape[0])
        k = 0
        while queue:
            cut, current_data = queue.pop()
            if cut.left == None:
                clustering[current_data] = k
                k += 1
            else:
                try:
                    left_bool = data[current_data,cut.coordinate] <= cut.threshold
                except IndexError:
                    raise ValueError("Input data dimensions do not match a Cut's split coordinate.")
                l_partition = current_data[left_bool]
                r_partition = current_data[~left_bool]
                if l_partition.size != 0:
                    queue.append((cut.left, l_partition))
                if r_partition.size != 0:
                    queue.append((cut.right, r_partition))

        return clustering
    
@dataclass(order=True)
class CutComp():
    """
    Stores information needed to compare promising splits.
    """
    cond: float 
    cut: Cut = field(compare=False)
    indices: np.ndarray | tuple[np.ndarray, np.ndarray] = field(compare=False)

class _Base_ExplainableTree(ABC):
    def __init__(self):
        self.tree = Tree()

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    def _order_current_data(self, data: np.ndarray, current_data: np.ndarray, i: int) -> np.ndarray:
        return current_data[data[current_data,i].argsort()]
    
    def _split(self, data: np.ndarray, current_data: np.ndarray, coordinate: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        left_data_bool = data[current_data, coordinate] <= threshold
        return current_data[left_data_bool], current_data[~left_data_bool]
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Clusters input data using the tree structure.
        See Tree.predict
        """
        return self.tree.predict(data)

@njit([(int32[:], int32[:], int64), (int64[:], int64[:], int64)], parallel=True)
def _local_counts(clustering: np.ndarray, current_data: np.ndarray, k: int) -> np.ndarray:
    local_counts = np.zeros(k, dtype=np.int32)
    for j in prange(k):
        local_counts[j] = (clustering[current_data] == j).sum()
    return local_counts

class RefMixin():
    def get_local_counts(self, clustering: np.ndarray, current_data: np.ndarray, k: int) -> np.ndarray:
        return _local_counts(clustering, current_data, k)
    
class GraphMixin():
    def best_data_point(self, metric: np.ndarray, inner_data: np.ndarray) -> int:
        # assert metric.size == inner_data.size
        # if not (np.isfinite(metric[:-1])).all():
        #     print(f"WARNING: {metric}")
        return np.argmin(np.where(
            (inner_data[:-1] < inner_data[1:]) # Cannot split two points with the same value.
             & np.isfinite(metric[:-1]),
            metric[:-1],
            np.inf))