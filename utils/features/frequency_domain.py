import numpy as np
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import moment, skew, kurtosis, entropy
from typing import List, Tuple



class SpectralFeatures:
    """
    A class for computing Fourier Transform or Power Spectral Density (PSD) and extracting various frequency domain features from a signal.

    Args:
    signal (array-like): The input signal for feature extraction.
    sample_rate (float): The sampling rate of the signal.

    Attributes:
    spectrum (array-like): The Fourier spectrum (or PSD) of the input signal.
    frequencies (array-like): The corresponding frequencies of the spectrum.
    """

    def __init__(self, signal: np.ndarray, sample_rate: float):
        self.signal = signal
        self.sample_rate = sample_rate
        # Compute the Power Spectral Density (PSD)
        self.frequencies, self.spectrum = welch(signal, fs=sample_rate)

    def mean(self) -> float:
        """
        Compute the mean of the magnitude spectrum (or PSD).
        """
        return np.mean(self.spectrum)


    def variance(self) -> float:
        """
        Compute the variance of the magnitude spectrum.
        """
        return np.var(np.abs(self.spectrum))

    def skewness(self) -> float:
        """
        Compute the skewness of the magnitude spectrum.
        """
        return skew(np.abs(self.spectrum))

    def kurt(self) -> float:
        """
        Compute the kurtosis of the magnitude spectrum.
        """
        return kurtosis(np.abs(self.spectrum))

    def energy(self) -> float:
        """
        Compute the total energy of the signal in the frequency domain.
        """
        return np.sum(np.abs(self.spectrum) ** 2)

    def frequency_bands(self, band_ranges: List[Tuple[float, float]]) -> List[float]:
        """
        Compute the energy within specified frequency bands.

        Args:
        band_ranges (list of tuples): List of tuples specifying frequency band ranges (e.g., [(low1, high1), (low2, high2)]).

        Returns:
        list: Energy in each specified frequency band.
        """
        energy_bands = []
        for band_range in band_ranges:
            low, high = band_range
            indices = np.where((self.frequencies >= low) & (self.frequencies <= high))
            energy_bands.append(np.sum(np.abs(self.spectrum[indices]) ** 2))
        return energy_bands

    def peak_frequency(self) -> float:
        """
        Find the frequency with the highest magnitude in the spectrum.
        """
        return self.frequencies[np.argmax(np.abs(self.spectrum))]

    def frequency_centroid(self) -> float:
        """
        Compute the frequency centroid of the spectrum.
        """
        return np.sum(np.abs(self.spectrum) * self.frequencies) / np.sum(np.abs(self.spectrum))

    def bandwidth(self) -> float:
        """
        Compute the bandwidth of the spectrum.
        """
        return np.sum(np.abs(self.spectrum) * (self.frequencies - self.frequency_centroid()) ** 2) / np.sum(np.abs(self.spectrum))

    def spectral_entropy(self) -> float:
        """
        Compute the spectral entropy of the magnitude spectrum.

        Returns:
        float: The spectral entropy.
        """
        p = np.abs(self.spectrum) / np.sum(np.abs(self.spectrum))
        return -np.sum(p * np.log2(p))

    def relative_power(self, band_range: Tuple[float, float]) -> float:
        """
        Compute the relative power within a specific frequency band.

        Args:
        band_range (tuple): Frequency band range (e.g., (low, high)).

        Returns:
        float: The relative power in the specified frequency band.
        """
        low, high = band_range
        indices = np.where((self.frequencies >= low) & (self.frequencies <= high))
        return np.sum(np.abs(self.spectrum[indices]) ** 2) / self.energy()

    def cross_frequency_ratios(self, band_ranges: List[Tuple[float, float]]) -> List[float]:
        """
        Compute cross-frequency ratios between specified frequency bands.

        Args:
        band_ranges (list of tuples): List of tuples specifying frequency band ranges (e.g., [(low1, high1), (low2, high2)]).

        Returns:
        list: Cross-frequency ratios between specified frequency bands.
        """
        ratios = []
        for i, band1 in enumerate(band_ranges):
            for j, band2 in enumerate(band_ranges):
                if i < j:
                    energy_band1 = self.relative_power(band1)
                    energy_band2 = self.relative_power(band2)
                    ratios.append(energy_band1 / energy_band2)
        return ratios
