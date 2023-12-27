import numpy as np
import itertools

from scipy import stats
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform

from typing import List, Tuple, Union

class StatisticalFeatures:
    def __init__(self, eeg_signal: List[float]):
        self.eeg_signal = np.array(eeg_signal)

    def mean(self) -> float:
        return np.mean(self.eeg_signal)

    def std_deviation(self) -> float:
        return np.std(self.eeg_signal)

    def variance(self) -> float:
        return np.var(self.eeg_signal)

    def mode(self) -> float:
        return float(stats.mode(self.eeg_signal).mode)

    def median(self) -> float:
        return np.median(self.eeg_signal)

    def skewness(self) -> float:
        return float(stats.skew(self.eeg_signal))

    def kurtosis(self) -> float:
        return float(stats.kurtosis(self.eeg_signal))

    def minimum(self) -> float:
        return np.min(self.eeg_signal)

    def maximum(self) -> float:
        return np.max(self.eeg_signal)

    def coefficient_of_variation(self) -> float:
        mean = self.mean()
        std_dev = self.std_deviation()
        if mean == 0:
            return 0  # To avoid division by zero if mean is zero
        return std_dev / mean

    def quartiles(self) -> Tuple[float, float]:
        q1, q3 = np.percentile(self.eeg_signal, [25, 75])
        return float(q1), float(q3)

    def inter_quartile_range(self) -> float:
        q1, q3 = self.quartiles()
        return q3 - q1
    

class AmplitudeFeatures:
    def __init__(self, eeg_signal: List[float]):
        self.eeg_signal = np.array(eeg_signal)

    def energy(self) -> float:
        """
        Calculate the energy of the EEG signal, which quantifies the signal's overall magnitude.
        Energy is computed by squaring each data point in the signal and summing all squared values.

        Returns:
            float: The energy of the EEG signal.
        """
        return np.sum(self.eeg_signal ** 2)

    def average_power(self) -> float:
        """
        Calculate the average power of the EEG signal, which measures its average strength.
        Average power is computed as the mean of squared values of the signal.

        Returns:
            float: The average power of the EEG signal.
        """
        return np.mean(self.eeg_signal ** 2)

    def root_mean_squared(self) -> float:
        """
        Calculate the root mean squared (RMS) value of the EEG signal, representing its typical amplitude.
        RMS is computed as the square root of the mean of squared values in the signal.

        Returns:
            float: The root mean squared value of the EEG signal.
        """
        return np.sqrt(np.mean(self.eeg_signal ** 2))

    def line_length(self) -> float:
        """
        Calculate the line length of the EEG signal, which quantifies the signal's variation.
        Line length is computed by summing the absolute differences between consecutive data points.

        Returns:
            float: The line length of the EEG signal.
        """
        diff_signal = np.diff(self.eeg_signal)
        return np.sum(np.abs(diff_signal))

    def zero_crossings(self) -> int:
        """
        Count the number of zero crossings in the EEG signal, indicating changes in polarity.
        Zero crossings are points where the signal changes from positive to negative or vice versa.

        Returns:
            int: The number of zero crossings in the EEG signal.
        """
        zero_crossings = np.where(np.diff(np.sign(self.eeg_signal)))[0]
        return len(zero_crossings)

    def local_extrema(self) -> int:
        """
        Count the number of local extrema (maxima and minima) in the EEG signal.
        Local extrema are points representing peaks (maxima) and valleys (minima) in the signal.

        Returns:
            int: The number of local extrema in the EEG signal.
        """
        extrema = np.where((np.diff(np.sign(np.diff(self.eeg_signal))) > 0) | (np.diff(np.sign(np.diff(self.eeg_signal))) < 0))[0]
        return len(extrema)



class HjorthParameters:
    def __init__(self, eeg_signal):
        self.eeg_signal = eeg_signal
        self.first_derivative = np.diff(eeg_signal)
        self.second_derivative = np.diff(self.first_derivative)
        
    def activity(self):
        return np.var(self.eeg_signal)

    def mobility(self):
        return np.sqrt(np.var(self.first_derivative) / self.activity())

    def complexity(self):
        return np.sqrt(np.var(self.second_derivative) / np.var(self.first_derivative))



class EntropyFeatures:
    def __init__(self, eeg_signal):
        self.eeg_signal = np.array(eeg_signal)


    def _distance(self, x, y):
        """
        Calculate the Chebyshev distance between two vectors.

        Args:
            x (numpy.ndarray): The first vector.
            y (numpy.ndarray): The second vector.

        Returns:
            float: The Chebyshev distance between vectors x and y.
        """
        return np.max(np.abs(x - y))

    def shannon_entropy(self):
        """
        Calculate the Shannon Entropy of the EEG signal.

        Returns:
            float: Shannon Entropy value.
        """
        unique_values, counts = np.unique(self.eeg_signal, return_counts=True)
        probabilities = counts / len(self.eeg_signal)
        return -np.sum(probabilities * np.log2(probabilities))

    def approximate_entropy(self, m=2, r=0.2):
        """
        Calculate the Approximate Entropy of the EEG signal.
        Args:
            m (int): Length of compared runs (default is 2).
            r (float): Tolerance threshold (default is 0.2).

        Returns:
            float: Approximate Entropy value.
        """
        N = len(self.eeg_signal)       
        phi = lambda m: np.mean([np.log(np.sum([self._distance(self.eeg_signal[i:i+m], self.eeg_signal[j:j+m]) <= r for j in range(N - m + 1)])) for i in range(N - m + 1)])
        return phi(m) - phi(m + 1)
    

    def sample_entropy(self, m=2, r=0.2):
        """
        Calculate the Sample Entropy of the EEG signal.

        Args:
            m (int): Length of compared runs (default is 2).
            r (float): Tolerance threshold (default is 0.2).

        Returns:
            float: Sample Entropy value.
        """
        N = len(self.eeg_signal)
        phi = lambda m: sum(self._distance(self.eeg_signal[i:i+m], self.eeg_signal[j:j+m]) < r for i in range(N - m) for j in range(i + 1, N - m) if i != j)
        return -np.log(phi(m + 1) / phi(m)) if phi(m) > 0 else 0


    def fuzzy_entropy(self, m=2, r=0.2):
        """
        Calculate the Fuzzy Entropy of the EEG signal.

        Args:
            m (int): Length of compared runs (default is 2).
            r (float): Tolerance threshold (default is 0.2).

        Returns:
            float: Fuzzy Entropy value.
        """

        N = len(self.eeg_signal)
        phi = lambda m: np.mean([np.exp([(-((self._distance(self.eeg_signal[i:i+m], self.eeg_signal[j:j+m]) ** 2) / r)) for j in range(N - m + 1) if j != i]) for i in range(N - m + 1)])
        return np.log(phi(m) / phi(m + 1))
    

    def distribution_entropy(self, m=2, num_bins=10):
        """
        Calculate the Distribution Entropy of the EEG signal.

        Args:
            m (int): Length of compared runs (default is 2).
            num_bins (int): Number of bins for histogram (default is 10).

        Returns:
            float: Distribution Entropy value.
        """

        N = len(self.eeg_signal)       
        distance_matrix = lambda m: [[self._distance(self.eeg_signal[i:i+m], self.eeg_signal[j:j+m]) for j in range(N - m + 1)] for i in range(N - m + 1)]
        # distances = np.array([np.array(distance_matrix(m))[i, j] for j in range(i, len(distance_matrix(m))) for i in range(len(distance_matrix(m)))])
        distances = np.array([np.array(distance_matrix(m))[i, j] for i in range(len(distance_matrix(m))) for j in range(i, len(distance_matrix(m)))])
        distances = distances[distances != 0.0] # drop the zeros from 
        
        hist, _  = np.histogram(distances, bins=num_bins, density=False)
        probabilities = hist / np.sum(hist)
        return stats.entropy(probabilities, base=2) /np.log(num_bins)
    

    def permutation_entropy(self, m=3):
        """
        Calculate the Permutation Entropy of the EEG signal.

        Args:
            m (int): Embedding dimension (default is 3).

        Returns:
            float: Permutation Entropy value.
        """
        n = len(self.eeg_signal)
        permutations = list(itertools.permutations(range(m)))
        num_permutations = len(permutations)

        data_matrix = np.zeros((n - m + 1, m))
        for i in range(n - m + 1):
            data_matrix[i, :] = self.eeg_signal[i:i + m]

        sorted_indices = np.argsort(data_matrix, axis=1)
        permutation_indices = np.zeros(n - m + 1)

        for i, perm in enumerate(permutations):
            for j in range(n - m + 1):
                if np.array_equal(sorted_indices[j, :], perm):
                    permutation_indices[j] = i

        counts = np.bincount(permutation_indices.astype(int))
        probabilities = counts / (n - m + 1)

        entropy_value = -1 * np.sum(probabilities * np.log(probabilities))
        return entropy_value
    

    def weighted_permutation_entropy(self, m=3):
        """
        Calculate the Weighted Permutation Entropy of the EEG signal.

        Args:
            m (int): Embedding dimension (default is 3).

        Returns:
            float: Weighted Permutation Entropy value.
        """
        n = len(self.eeg_signal)
        permutations = list(itertools.permutations(range(m)))
        num_permutations = len(permutations)

        data_matrix = np.zeros((n - m + 1, m))
        for i in range(n - m + 1):
            data_matrix[i, :] = self.eeg_signal[i:i + m]

        sorted_indices = np.argsort(data_matrix, axis=1)
        permutation_indices = np.zeros(n - m + 1)

        for i, perm in enumerate(permutations):
            for j in range(n - m + 1):
                if np.array_equal(sorted_indices[j, :], perm):
                    permutation_indices[j] = i

        # Calculate the variance of each template vector
        template_variances = np.var(data_matrix, axis=1)   

        probabilities = [np.sum(template_variances[permutation_indices==perm_idx]) / np.sum(template_variances) for perm_idx in np.unique(permutation_indices)]
        entropy_value = -1.0 * np.sum(probabilities * np.log(probabilities))
        return entropy_value