import numpy as np
from scipy.signal import butter, lfilter, resample

class Preprocessor:
    def __init__(self):
        pass
    def z_score_normalize(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Z-score normalize the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.

        Returns:
        - normalized_signal (np.ndarray): Z-score normalized EEG signal.
        """
        mean = np.mean(eeg_signal)
        std = np.std(eeg_signal)
        normalized_signal = (eeg_signal - mean) / std
        return normalized_signal

    def min_max_scale(self, eeg_signal: np.ndarray, feature_range=(0, 1)) -> np.ndarray:
        """
        Perform min-max scaling on the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - feature_range (tuple): Desired range for scaled values (default: [0, 1]).

        Returns:
        - scaled_signal (np.ndarray): Min-max scaled EEG signal within the specified range.
        """
        min_val, max_val = feature_range
        min_signal = np.min(eeg_signal)
        max_signal = np.max(eeg_signal)
        scaled_signal = ((eeg_signal - min_signal) / (max_signal - min_signal)) * (max_val - min_val) + min_val
        return scaled_signal

    def high_pass(self, eeg_signal: np.ndarray, cutoff_freq: float, sampling_rate: float) -> np.ndarray:
        """
        Apply a high-pass Butterworth filter to the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - cutoff_freq (float): Cutoff frequency for the high-pass filter.
        - sampling_rate (float): Sampling rate of the EEG signal.

        Returns:
        - filtered_signal (np.ndarray): High-pass filtered EEG signal.
        """
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        filtered_signal = lfilter(b, a, eeg_signal)
        return filtered_signal

    def low_pass(self, eeg_signal: np.ndarray, cutoff_freq: float, sampling_rate: float) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - cutoff_freq (float): Cutoff frequency for the low-pass filter.
        - sampling_rate (float): Sampling rate of the EEG signal.

        Returns:
        - filtered_signal (np.ndarray): Low-pass filtered EEG signal.
        """
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        filtered_signal = lfilter(b, a, eeg_signal)
        return filtered_signal

    def bandpass(self, eeg_signal: np.ndarray, low_cutoff_freq: float, high_cutoff_freq: float, sampling_rate: float) -> np.ndarray:
        """
        Apply a band-pass Butterworth filter to the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - low_cutoff_freq (float): Low cutoff frequency for the band-pass filter.
        - high_cutoff_freq (float): High cutoff frequency for the band-pass filter.
        - sampling_rate (float): Sampling rate of the EEG signal.

        Returns:
        - filtered_signal (np.ndarray): Band-pass filtered EEG signal.
        """
        nyquist_freq = 0.5 * sampling_rate
        low_normal_cutoff = low_cutoff_freq / nyquist_freq
        high_normal_cutoff = high_cutoff_freq / nyquist_freq
        b, a = butter(4, [low_normal_cutoff, high_normal_cutoff], btype='band', analog=False)
        filtered_signal = lfilter(b, a, eeg_signal)
        return filtered_signal

    def notch(self, eeg_signal: np.ndarray, notch_freq: float, quality_factor: float, sampling_rate: float) -> np.ndarray:
        """
        Apply a notch filter to remove a specific frequency component from the EEG signal.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - notch_freq (float): Frequency to be notched out.
        - quality_factor (float): Quality factor (Q) of the notch filter.
        - sampling_rate (float): Sampling rate of the EEG signal.

        Returns:
        - filtered_signal (np.ndarray): EEG signal with the specified frequency notched out.
        """
        nyquist_freq = 0.5 * sampling_rate
        normal_freq = notch_freq / nyquist_freq
        b, a = butter(4, normal_freq, btype='bandstop', analog=False, fs=sampling_rate)
        filtered_signal = lfilter(b, a, eeg_signal)
        return filtered_signal

    def resample_signal(self, eeg_signal: np.ndarray, target_sampling_rate: float) -> np.ndarray:
        """
        Resample the EEG signal to a new target sampling rate.

        Args:
        - eeg_signal (np.ndarray): Input EEG signal.
        - target_sampling_rate (float): Desired target sampling rate.

        Returns:
        - resampled_signal (np.ndarray): Resampled EEG signal at the target sampling rate.
        """
        current_sampling_rate = len(eeg_signal)
        resampled_signal = resample(eeg_signal, int(len(eeg_signal) * target_sampling_rate / current_sampling_rate))
        return resampled_signal
