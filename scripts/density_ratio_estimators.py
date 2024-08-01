from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
import numpy as np

class DensityRatioEstimator(ABC):
    @staticmethod
    def check_array_type_shape(array: np.ndarray, array_name: str) -> None:
        """
        Check whether an array meets various type and shape criteria.

        :param array: NumPy array of shape (n, d).
        :param array_name: Name of array.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{array_name} must be a NumPy array")
        if array.ndim != 2:
            raise ValueError(f"{array_name} must be 2-dimensional")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{array_name} must have at least one row and at least one column")

    @staticmethod
    def check_array_col_counts(array1: np.ndarray, array2: np.ndarray) -> None:
        """
        Check whether two arrays have the same number of columns.

        :param array1: NumPy array of shape (n1, d1).
        :param array2: NumPy array of shape (n2, d2).
        """
        if array1.shape[1] != array2.shape[1]:
            raise ValueError("arrays must have the same number of columns")

    @abstractmethod
    def fit(self, x_numer: np.ndarray, x_denom: np.ndarray, **kwargs) -> None:
        """
        Construct a density ratio estimator from the given data.

        :param x_numer: NumPy array of shape (n_numer, d) whose rows are samples from the numerator density.
        :param x_denom: NumPy array of shape (n_denom, d) whose rows are samples from the denominator density.
        :param kwargs: Additional keyword arguments for constructing the estimator.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate the value of a density ratio at specified points.

        :param x: NumPy array of shape (n, d) whose rows are the points at which the density ratio should be estimated.
        :return: NumPy array of shape (n,) containing the estimated values of the density ratio.
        """
        pass

class KNNDensityRatioEstimator(DensityRatioEstimator):
    def fit(self, x_numer: np.ndarray, x_denom: np.ndarray, k: int) -> None:
        """
        Construct a density ratio estimator from the given data that is based on kth nearest neighbors.

        :param x_numer: NumPy array of shape (n_numer, d) whose rows are samples from the numerator density.
        :param x_denom: NumPy array of shape (n_denom, d) whose rows are samples from the denominator density.
        :param k: Integer specifying which nearest neighbors to use.
        """
        DensityRatioEstimator.check_array_type_shape(x_numer, "x_numer")
        DensityRatioEstimator.check_array_type_shape(x_denom, "x_denom")
        DensityRatioEstimator.check_array_col_counts(x_numer, x_denom)
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 1:
            raise ValueError("k must be positive")
        if k > x_numer.shape[0] or k > x_denom.shape[0]:
            raise ValueError("k must be less than or equal to both sample sizes")
        self.x_numer = x_numer
        self.x_denom = x_denom
        self.k = k
        self.d = x_numer.shape[1]
        self.knns_finder = NearestNeighbors(n_neighbors=k)

    def _get_dists_to_knn(self, x: np.ndarray, sample_type: str) -> np.ndarray:
        """
        Compute the distance between each given point and its kth nearest neighbor in the specified sample.

        :param x: NumPy array of shape (n, d) whose rows are the points of interest.
        :return: String specifying whether to use the numerator sample or the denominator sample.
        """
        if sample_type == "numer":
            sample = self.x_numer
        elif sample_type == "denom":
            sample = self.x_denom
        else:
            raise ValueError("sample_type should be either 'numer' or 'denom'")
        dists_to_knns, _ = self.knns_finder.fit(sample).kneighbors(x)
        return dists_to_knns[:, self.k - 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate the value of a density ratio at specified points.

        :param x: NumPy array of shape (n, d) whose rows are the points at which the density ratio should be estimated.
        :return: NumPy array of shape (n,) containing the estimated values of the density ratio.
        """
        dists_to_numer_knn = self._get_dists_to_knn(x, "numer")
        dists_to_denom_knn = self._get_dists_to_knn(x, "denom")
        return (dists_to_denom_knn / dists_to_numer_knn) ** self.d
