import pywt
import numpy as np
import matplotlib.pyplot as plt

from PyEMD import EMD
from scipy.signal import hilbert
from scipy.signal import spectrogram


class EEGSpectrogram:
    def __init__(self, signal, sampling_rate, window_length=256, overlap=128):
        """
        Initialize the EEGSpectrogram object.

        Parameters:
            signal (array-like): The EEG signal.
            sampling_rate (int): The sampling rate of the EEG signal (samples per second).
            window_length (int, optional): Length of the analysis window for the STFT (default: 256).
            overlap (int, optional): Overlap between successive windows (default: 128).
        """
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.overlap = overlap

    def compute_spectrogram(self):
        """
        Compute the spectrogram of the EEG signal.
        
        Returns:
            frequencies (array): The frequencies.
            times (array): The time points.
            spectrogram_data (array): The spectrogram data.
        """
        frequencies, times, Sxx = spectrogram(self.signal, fs=self.sampling_rate, window='hamming', nperseg=self.window_length, noverlap=self.overlap)
        return frequencies, times, 10 * np.log10(Sxx)  # Applying logarithmic scaling for visualization

    def plot_spectrogram(self):
        """
        Plot the spectrogram of the EEG signal.
        """
        frequencies, times, spectrogram_data = self.compute_spectrogram()

        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times, frequencies, spectrogram_data)
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('EEG Spectrogram')
        plt.show()


class WaveletTransform:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the WaveletTransform object.

        Parameters:
            signal (array-like): The signal on which to perform the wavelet transform.
            sampling_rate (int): The sampling rate of the signal (samples per second).
        """
        self.signal = signal
        self.sampling_rate = sampling_rate

    def compute_cwt(self, wavelet_name, scales):
        """
        Compute the Continuous Wavelet Transform (CWT) of the signal with a specified wavelet.

        Parameters:
            wavelet_name (str): Name of the wavelet to use (e.g., 'morl', 'gaus2').
            scales (array-like): The scales at which to compute the CWT.

        Returns:
            coefficients (array): The CWT coefficients.
            frequencies (array): The corresponding frequencies.
        """
        coefficients, frequencies = pywt.cwt(self.signal, scales, wavelet_name)
        return coefficients, frequencies

    def plot_cwt(self, wavelet_name, scales):
        """
        Plot the Continuous Wavelet Transform (CWT) of the signal with a specified wavelet.

        Parameters:
            wavelet_name (str): Name of the wavelet to use (e.g., 'morl', 'gaus2').
            scales (array-like): The scales at which to compute the CWT.
        """
        coefficients, frequencies = self.compute_cwt(wavelet_name, scales)

        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(coefficients), extent=[0, len(self.signal), min(scales), max(scales)], aspect='auto', cmap='coolwarm')
        plt.colorbar(label='Magnitude')
        plt.title(f'{wavelet_name.capitalize()} Continuous Wavelet Transform')
        plt.xlabel('Time')
        plt.ylabel('Scale/Frequency')
        plt.show()


class HilbertTransform:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the HilbertHuangTransform object.

        Parameters:
            signal (array-like): The signal on which to perform the HHT.
            sampling_rate (int): The sampling rate of the signal (samples per second).
        """
        self.signal = signal
        self.sampling_rate = sampling_rate

    def compute_hht(self):
        """
        Compute the Hilbert-Huang Transform (HHT) of the signal.

        Returns:
            imfs (list of arrays): The Intrinsic Mode Functions (IMFs) obtained from Empirical Mode Decomposition (EMD).
            inst_freqs (list of arrays): The instantaneous frequencies corresponding to each IMF.
        """
        emd = EMD()
        imfs = emd(self.signal)
        inst_freqs = [np.diff(np.unwrap(np.angle(hilbert(imf)))) / (2 * np.pi) for imf in imfs]
        return imfs, inst_freqs


class HilbertSpectrum:
    def __init__(self, eeg_data):
        self.signal = eeg_data
        self.emd = EMD()
        self.imfs = self.emd(self.signal)

    def compute_hilbert_spectrum(self):
        # Initialize arrays to store instantaneous frequency and amplitude
        instantaneous_frequency = []
        amplitude = []
        hilbert_spectrum = np.zeros(shape=self.imfs.shape)

        # Calculate instantaneous frequency and amplitude for each IMF
        for i, imf in enumerate(self.imfs):
            analytic_signal = hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency.append(np.gradient(instantaneous_phase) / (2 * np.pi))
            amplitude.append(np.abs(analytic_signal))
            hilbert_spectrum[i, :] = np.abs(analytic_signal)

        # Calculate the Hilbert Spectral Density (HSD)
        # This can be done by summing the squared amplitude over all IMFs
        hsd = np.sum(np.array(amplitude) ** 2, axis=0)

        return hilbert_spectrum, hsd

    def plot_hilbert_spectrum(self):
        hilbert_spectrum, hsd = self.compute_hilbert_spectrum()

        # Plot the Hilbert Spectrum
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(hilbert_spectrum)
        plt.colorbar(label='Hilbert Spectral Density')
        plt.title('Hilbert Spectrum')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.show()



