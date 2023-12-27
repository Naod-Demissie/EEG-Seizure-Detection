import sys
import numpy as np
from typing import List, Tuple


sys.path.append('/content/drive/MyDrive/EEG-Seizure-Detection')
from utils import read_lbl_file
from utils.preprocessing.convert_montage import convert_to_bipolar


def extract_segments(eeg_signal: np.ndarray, segment_duration: float, step_size: float, sampling_frequency: int) -> np.ndarray:
    """
    Extracts segments from an EEG signal with specified parameters.

    Args:
        eeg_signal (np.ndarray): The EEG signal data as a 1D numpy array.
        segment_duration (float): The duration of each segment in seconds.
        step_size (float): The step size between consecutive segments in seconds.
        sampling_frequency (int): The sampling frequency of the EEG signal in Hz.

    Returns:
        List[np.ndarray]: A list of numpy arrays, where each array represents an EEG segment.
    """
    segment_length = int(segment_duration * sampling_frequency)
    step_size_samples = int(step_size * sampling_frequency)
    
    eeg_segments = []
    
    for i in range(0, len(eeg_signal) - segment_length + 1, step_size_samples):
        segment = eeg_signal[i:i + segment_length]
        
        # Pad the segment if there's not enough data left
        if len(segment) < segment_length:
            padding = segment_length - len(segment)
            segment = np.concatenate((segment, np.zeros(padding)))
        eeg_segments.append(segment)

    return np.array(eeg_segments)



def return_data_segment(edf_path: str, segment_duration: float = 1.0, step_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts EEG segments and their corresponding labels from an EDF file with specified parameters.

    Args:
        edf_path (str): The path to the EDF file.
        segment_duration (float, optional): The duration of each EEG segment in seconds (default is 1.0).
        step_size (float, optional): The step size between consecutive EEG segments in seconds (default is 1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The EEG segments as a 2D numpy array.
            - The segment labels as a 1D numpy array, where 1.0 represents 'seiz' and 0.0 represents 'bckg'.
    """
    bipolar_raw = convert_to_bipolar(edf_path)
    bipolar_data = bipolar_raw.get_data()

    sfreq = bipolar_raw.info['sfreq']

    lbl_path = edf_path.replace('.edf', '.lbl_bi')
    lbl_dicts = read_lbl_file(lbl_path)

    segment_labels = []
    eeg_segments = []
    for chnl_idx, chnl in enumerate(bipolar_data):
        for lbl_dict in lbl_dicts[chnl_idx]:
            (start_time, stop_time), cls_code = lbl_dict
            start_idx, stop_idx = int(start_time * sfreq), int(stop_time * sfreq)

            signal = chnl[start_idx: stop_idx]
            chnl_segments = extract_segments(
                eeg_signal=signal, 
                segment_duration=segment_duration, 
                step_size=step_size, 
                sampling_frequency=sfreq
            )

            eeg_segments.append(chnl_segments)
            segment_labels.append([{'bckg': 0.0, 'seiz': 1.0}[cls_code] for _ in range(len(chnl_segments))])

    eeg_segments = np.vstack(eeg_segments)
    segment_labels = np.hstack(segment_labels)
    return eeg_segments, segment_labels, sfreq



# def return_data_segment(edf_path, segment_duration=1.0, step_size=1.0):
#     bipolar_raw = convert_to_bipolar(edf_path)
#     bipolar_data = bipolar_raw.get_data()

#     sfreq = bipolar_raw.info['sfreq']

#     lbl_path = edf_path.replace('.edf', '.lbl_bi')
#     lbl_dicts = read_lbl_file(lbl_path)

#     segment_labels = []
#     eeg_segments = []
#     for chnl_idx, chnl in enumerate(bipolar_data):
#         for lbl_dict in lbl_dicts[chnl_idx]:
#             (start_time, stop_time), cls_code = lbl_dict
#             start_idx, stop_idx = int(start_time * sfreq), int(stop_time * sfreq)

#             signal = chnl[start_idx: stop_idx]
#             chnl_segments = extract_segments(eeg_signal=signal, segment_duration=segment_duration, step_size=step_size, sampling_frequency=sfreq)

#             eeg_segments.append(chnl_segments)
#             segment_labels.append([{'bckg': 0.0, 'seiz':1.0}[cls_code] for _ in range(len(chnl_segments))])

#     eeg_segments = np.vstack(eeg_segments)
#     segment_labels = np.hstack(segment_labels)
#     return eeg_segments, segment_labels