from abc import ABC, abstractmethod
from densratio import densratio
from sklearn.neighbors import NearestNeighbors
from typing import Tuple
import math
import numpy as np
import warnings

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
    NUMER = "numer"
    DENOM = "denom"

    def __init__(self, k_numer: int, k_denom: int) -> None:
        """
        Initialize a KNNDensityRatioEstimator instance.

        :param k_numer: Integer specifying how many nearest neighbors to use from the array of numerator samples.
        :param k_denom: Integer specifying how many nearest neighbors to use from the array of denominator samples.
        """
        for arg, val in {"k_numer": k_numer, "k_denom": k_denom}.items():
            if not isinstance(val, int):
                raise TypeError(f"{arg} must be an integer")
            if val <= 0:
                raise ValueError(f"{arg} must be positive")
        self.k_numer = k_numer
        self.k_denom = k_denom


    def fit(self, x_numer: np.ndarray, x_denom: np.ndarray) -> None:
        """
        Construct a density ratio estimator from the given arrays of numerator and denominator samples.

        :param x_numer: NumPy array of shape (n_numer, d) whose rows are samples from the numerator density.
        :param x_denom: NumPy array of shape (n_denom, d) whose rows are samples from the denominator density.
        """
        DensityRatioEstimator.check_array_type_shape(x_numer, "x_numer")
        DensityRatioEstimator.check_array_type_shape(x_denom, "x_denom")
        DensityRatioEstimator.check_array_col_counts(x_numer, x_denom)
        if x_numer.shape[0] < self.k_numer:
            self.k_numer = x_numer.shape[0]
            warnings.warn("number of numerator samples is less than number of numerator nearest neighbors")
        if x_denom.shape[0] < self.k_denom:
            self.k_denom = x_denom.shape[0]
            warnings.warn("number of denominator samples is less than number of denominator nearest neighbors")
        self.x_numer = x_numer
        self.x_denom = x_denom
        self.d = x_numer.shape[1]
        self.numer_knns_finder = NearestNeighbors(n_neighbors=self.k_numer)
        self.denom_knns_finder = NearestNeighbors(n_neighbors=self.k_denom)

    def _get_dist_to_knn(self, x: np.ndarray, sample_type: str) -> np.ndarray:
        """
        Compute the distance between each point in a given array and its kth nearest neighbor in the specified sample.

        :param x: NumPy array of shape (n, d) whose rows are the points of interest.
        :param sample_type: String ('numer' or 'denom') specifying whether to use the numerator or the denominator sample.
        :return: NumPy array of shape (n,) containing the distance between each point and its kth nearest neighbor.
        """
        if sample_type == self.NUMER:
            sample = self.x_numer
            knns_finder = self.numer_knns_finder
            k = self.k_numer
        elif sample_type == self.DENOM:
            sample = self.x_denom
            knns_finder = self.denom_knns_finder
            k = self.k_denom
        else:
            raise ValueError(f"sample_type should be either '{self.NUMER}' or '{self.DENOM}'")
        dists_to_knns, _ = knns_finder.fit(sample).kneighbors(x)
        return dists_to_knns[:, k - 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate the values of a density ratio at points in a given array.

        :param x: NumPy array of shape (n, d) whose rows are the points at which the density ratio should be estimated.
        :return: NumPy array of shape (n,) containing the estimated values of the density ratio.
        """
        dists_to_numer_knn = self._get_dist_to_knn(x, self.NUMER)
        dists_to_denom_knn = self._get_dist_to_knn(x, self.DENOM)
        return (dists_to_denom_knn / dists_to_numer_knn) ** self.d

class KNN2DensityRatioEstimator(DensityRatioEstimator):
    def __init__(self, k: int) -> None:
        """
        Initialize a KNN2DensityRatioEstimator instance.

        :param k: Integer specifying how many nearest neighbors to use from the arrays of numerator and denominator samples.
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def fit(self, x_numer: np.ndarray, x_denom: np.ndarray) -> None:
        """
        Construct a density ratio estimator from the given arrays of numerator and denominator samples.

        :param x_numer: NumPy array of shape (n_numer, d) whose rows are samples from the numerator density.
        :param x_denom: NumPy array of shape (n_denom, d) whose rows are samples from the denominator density.
        """
        DensityRatioEstimator.check_array_type_shape(x_numer, "x_numer")
        DensityRatioEstimator.check_array_type_shape(x_denom, "x_denom")
        DensityRatioEstimator.check_array_col_counts(x_numer, x_denom)
        self.x_numer = x_numer
        self.n_numer = x_numer.shape[0]
        self.x_denom = x_denom
        self.n_denom = x_denom.shape[0]
        self.knns_finder = NearestNeighbors(n_neighbors=self.k)

    def _get_dists_to_knns(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_combined = np.concatenate((self.x_numer, self.x_denom))
        dists_to_knns, indices = self.knns_finder.fit(x_combined).kneighbors(x)
        numer_flags = indices <= self.n_numer
        return dists_to_knns, numer_flags

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate the values of a density ratio at points in a given array.

        :param x: NumPy array of shape (n, d) whose rows are the points at which the density ratio should be estimated.
        :return: NumPy array of shape (n,) containing the estimated values of the density ratio.
        """
        dists_to_knns, numer_flags = self._get_dists_to_knns(x)
        n = x.shape[0]
        numer_vals = np.array([np.sum(np.exp(-dists_to_knns[i][numer_flags[i]])) for i in range(n)])
        denom_vals = np.array([np.sum(np.exp(-dists_to_knns[i][~numer_flags[i]])) for i in range(n)])
        k_numer = math.ceil(self.k * (self.n_numer / (self.n_numer + self.n_denom)))
        k_denom = self.k - k_numer
        return (numer_vals / k_numer) / (numer_vals / k_numer + denom_vals / k_denom) # Not normalized

class RuLSIFDensityRatioEstimator(DensityRatioEstimator):
    def __init__(self, sigma_range="auto", lambda_range="auto", kernel_num=100, verbose=True) -> None:
        self.sigma_range = sigma_range
        self.lambda_range = lambda_range
        self.kernel_num = kernel_num
        self.verbose = verbose

    def fit(self, x_numer: np.ndarray, x_denom: np.ndarray) -> None:
        self.estimator = densratio(
            x_numer, x_denom,
            sigma_range=self.sigma_range, lambda_range=self.lambda_range,
            kernel_num=self.kernel_num,
            verbose=self.verbose
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.compute_density_ratio(x)
