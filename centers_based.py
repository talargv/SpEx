from base_classes import _Base_ExplainableTree, RefMixin, Cut
import numpy as np

def _get_mistakes(left_side_data, centers_mask, left_side_centers_mask, local_counts):
    res = left_side_data[centers_mask & (~left_side_centers_mask)].sum() + (local_counts - left_side_data)[left_side_centers_mask].sum()
    return res

def _imm(left_side_data, centers_mask, left_side_centers_mask, local_counts, *args):
    return _get_mistakes(left_side_data, centers_mask, left_side_centers_mask, local_counts)

def _emn(left_side_data, centers_mask, left_side_centers_mask, local_counts, num_current_centers):
    left_clusters = left_side_centers_mask.sum()
    return _get_mistakes(left_side_data, centers_mask, left_side_centers_mask, local_counts) / min(left_clusters, num_current_centers - left_clusters)

class CentersBased(_Base_ExplainableTree, RefMixin):
    def __init__(self, algorithm="imm"):
        """
        Implements the IMM and EMN algorithms.
        """
        super().__init__()
        if algorithm == 'imm':    
            self.algorithm = _imm
        elif algorithm == 'emn':
            self.algorithm = _emn
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    def single_partition(self, data: np.ndarray, clustering: np.ndarray, centers: np.ndarray, current_data: np.ndarray, current_centers: np.ndarray) -> tuple[float, int, float]:
        algo = self.algorithm
        best_cond, best_index, best_threshold = np.inf, 0, -np.inf 
        k = centers.shape[0]

        if current_centers.size <= 1:
            return best_cond, best_index, best_threshold

        local_counts = self.get_local_counts(clustering, current_data, k)

        left_side_data = np.zeros(k)
        left_side_centers_mask = np.zeros(k, dtype='bool')
        centers_mask = np.zeros(k, dtype='bool')
        centers_mask[current_centers] = True
        
        for i in range(data.shape[1]):
            data_ordering = self._order_current_data(data, current_data, i)
            centers_ordering = self._order_current_data(centers, current_centers, i)

            left_side_data.fill(0)
            left_side_centers_mask.fill(False)

            center_index, data_index = 0, 0
            left_side_data[clustering[data_ordering[data_index]]] = 1
            left_side_centers_mask[centers_ordering[center_index]] = True
            while center_index < current_centers.size - 1 and data_index < current_data.size - 1:
                data_threshold, center_threshold = data[data_ordering[data_index],i], centers[centers_ordering[center_index], i]
                data_threshold_next, center_threshold_next = data[data_ordering[data_index+1],i], centers[centers_ordering[center_index+1], i]
                if (data_threshold == data_threshold_next) or (data_threshold <= center_threshold and data_threshold_next <= center_threshold) :
                    data_index += 1
                    left_side_data[clustering[data_ordering[data_index]]] += 1
                elif (center_threshold == center_threshold_next) or (data_threshold >= center_threshold and data_threshold >= center_threshold_next):
                    center_index += 1
                    left_side_centers_mask[centers_ordering[center_index]] = True
                else:
                    cond = algo(left_side_data, centers_mask, left_side_centers_mask, local_counts, current_centers.size)

                    if np.isfinite(cond) and cond < best_cond:
                        best_cond, best_index, best_threshold = cond, i, max(data_threshold, center_threshold)
                    
                    if data_threshold_next <= center_threshold_next:
                        data_index += 1
                        left_side_data[clustering[data_ordering[data_index]]] += 1
                    else:
                        center_index += 1
                        left_side_centers_mask[centers_ordering[center_index]] = True
        if not np.isfinite(best_cond):
            pass
        return best_cond, best_index, best_threshold

    def train(self, data: np.ndarray, clustering: np.ndarray, centers: np.ndarray):
        """
        Trains a tree with k leaves on a given dataset and a reference clustering.

        Args:
            data: np.ndarray
                The full dataset.
                Dimensions should be (num samples, sample dimension).
            clustering: np.ndarray
                The full clustering of the data.
                Dimensions should be (num samples,).
                Assumes centers[clustering[j]] = center of cluster that x_j belongs to
            centers: np.ndarray
                The full centers of the clusters.
                Dimensions should be (num centers, sample dimension).
        """
        current_data, current_centers =  np.arange(data.shape[0]), np.arange(centers.shape[0])
        queue = []

        _, best_index, best_threshold = self.single_partition(data, clustering, centers, current_data, current_centers)
        self.tree.root.coordinate = best_index
        self.tree.root.threshold = best_threshold
        queue.append((self.tree.root, current_data, current_centers))

        for _ in range(centers.shape[0]-1):
            cut, current_data, current_centers = queue.pop()

            left_data, right_data = self._split(data, current_data, cut.coordinate, cut.threshold)
            left_centers, right_centers = self._split(centers, current_centers, cut.coordinate, cut.threshold)

            l_best_cond, l_best_index, l_best_threshold = self.single_partition(data, clustering, centers, left_data, left_centers)
            r_best_cond, r_best_index, r_best_threshold = self.single_partition(data, clustering, centers, right_data, right_centers)

            cut_left = Cut(l_best_index, l_best_threshold)
            cut_right = Cut(r_best_index, r_best_threshold)

            cut.left = cut_left
            cut.right = cut_right

            if np.isfinite(l_best_cond):
                queue.append((cut_left, left_data, left_centers))
            if np.isfinite(r_best_cond):
                queue.append((cut_right, right_data, right_centers))