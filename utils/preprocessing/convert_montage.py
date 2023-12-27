import mne
import numpy as np

def bipolar_name_mapping(file_path: str, montage_type:str) -> dict:
# def bipolar_name_mapping(file_path: str, montage_type:str = None) -> dict:
    '''
    maps the custom naming used in the dataset to the conventional
    naming for bipolar montage

    Parameter
    ---------
    file_path: path to the `*.edf` file. `montage_type` is assumed
    to be included in this path

    montage_type: either of `01_tcp_ar`, `02_tcp_le` or
    `03_tcp_ar_a` values. This value decide which channels gets
    included name mapping.

    Returns
    -------
    a dictionary of names with standard naming as keys and
    dataset naming as values.

    '''
    if montage_type is None:
        montage_type = file_path.split('/')[-5]

    if montage_type == '01_tcp_ar':
        return {
            'FP1-F7': 'EEG FP1-REF--EEG F7-REF',
            'F7-T3':  'EEG F7-REF--EEG T3-REF',
            'T3-T5':  'EEG T3-REF--EEG T5-REF',
            'T5-O1':  'EEG T5-REF--EEG O1-REF',
            'FP2-F8': 'EEG FP2-REF--EEG F8-REF',
            'F8-T4':  'EEG F8-REF--EEG T4-REF',
            'T4-T6':  'EEG T4-REF--EEG T6-REF',
            'T6-O2':  'EEG T6-REF--EEG O2-REF',
            'A1-T3':  'EEG A1-REF--EEG T3-REF',
            'T3-C3':  'EEG T3-REF--EEG C3-REF',
            'C3-CZ':  'EEG C3-REF--EEG CZ-REF',
            'CZ-C4':  'EEG CZ-REF--EEG C4-REF',
            'C4-T4':  'EEG C4-REF--EEG T4-REF',
            'T4-A2':  'EEG T4-REF--EEG A2-REF',
            'FP1-F3': 'EEG FP1-REF--EEG F3-REF',
            'F3-C3':  'EEG F3-REF--EEG C3-REF',
            'C3-P3':  'EEG C3-REF--EEG P3-REF',
            'P3-O1':  'EEG P3-REF--EEG O1-REF',
            'FP2-F4': 'EEG FP2-REF--EEG F4-REF',
            'F4-C4':  'EEG F4-REF--EEG C4-REF',
            'C4-P4':  'EEG C4-REF--EEG P4-REF',
            'P4-O2':  'EEG P4-REF--EEG O2-REF'
        }

    elif montage_type == '02_tcp_le':
        return {
            'P1-F7':'EEG FP1-LE--EEG F7-LE',
            'F7-T3': 'EEG F7-LE--EEG T3-LE',
            'T3-T5': 'EEG T3-LE--EEG T5-LE',
            'T5-O1': 'EEG T5-LE--EEG O1-LE',
            'FP2-F8': 'EEG FP2-LE--EEG F8-LE',
            'F8-T4': 'EEG F8-LE--EEG T4-LE',
            'T4-T6': 'EEG T4-LE--EEG T6-LE',
            'T6-O2': 'EEG T6-LE--EEG O2-LE',
            'A1-T3': 'EEG A1-LE--EEG T3-LE',
            'T3-C3': 'EEG T3-LE--EEG C3-LE',
            'C3-CZ': 'EEG C3-LE--EEG CZ-LE',
            'CZ-C4': 'EEG CZ-LE--EEG C4-LE',
            'C4-T4': 'EEG C4-LE--EEG T4-LE',
            'T4-A2': 'EEG T4-LE--EEG A2-LE',
            'P1-F3': 'EEG FP1-LE--EEG F3-LE',
            'F3-C3': 'EEG F3-LE--EEG C3-LE',
            'C3-P3': 'EEG C3-LE--EEG P3-LE',
            'P3-O1': 'EEG P3-LE--EEG O1-LE',
            'P2-F4': 'EEG FP2-LE--EEG F4-LE',
            'F4-C4': 'EEG F4-LE--EEG C4-LE',
            'C4-P4': 'EEG C4-LE--EEG P4-LE',
            'P4-O2': 'EEG P4-LE--EEG O2-LE'
        }

    elif montage_type == '03_tcp_ar_a':
        return {
            'FP1-F7': 'EEG FP1-REF--EEG F7-REF',
            'F7-T3':  'EEG F7-REF--EEG T3-REF',
            'T3-T5':  'EEG T3-REF--EEG T5-REF',
            'T5-O1':  'EEG T5-REF--EEG O1-REF',
            'FP2-F8': 'EEG FP2-REF--EEG F8-REF',
            'F8-T4':  'EEG F8-REF--EEG T4-REF',
            'T4-T6':  'EEG T4-REF--EEG T6-REF',
            'T6-O2':  'EEG T6-REF--EEG O2-REF',
            'T3-C3':  'EEG T3-REF--EEG C3-REF',
            'C3-CZ':  'EEG C3-REF--EEG CZ-REF',
            'CZ-C4':  'EEG CZ-REF--EEG C4-REF',
            'C4-T4':  'EEG C4-REF--EEG T4-REF',
            'FP1-F3': 'EEG FP1-REF--EEG F3-REF',
            'F3-C3':  'EEG F3-REF--EEG C3-REF',
            'C3-P3':  'EEG C3-REF--EEG P3-REF',
            'P3-O1':  'EEG P3-REF--EEG O1-REF',
            'FP2-F4': 'EEG FP2-REF--EEG F4-REF',
            'F4-C4':  'EEG F4-REF--EEG C4-REF',
            'C4-P4':  'EEG C4-REF--EEG P4-REF',
            'P4-O2':  'EEG P4-REF--EEG O2-REF'
        }



def convert_to_bipolar(file_path: str, montage_type: str = None) -> mne.io.RawArray:
    # convert unipolar montage to bipolar montage for plotting and model training
    bipolar_name_map = bipolar_name_mapping(file_path, montage_type)

    raw_eeg = mne.io.read_raw_edf(file_path, verbose=False, preload=True)

    bipolar_chn_names = []
    bipolar_chn_val = []
    for bipolar_chnl_key, bipolar_chnl_val in bipolar_name_map.items():
        ch_cathode, ch_anode =  bipolar_chnl_val.split('--')
        bipolar_chn_names.append(bipolar_chnl_key)
        bipolar_chn_val.append(raw_eeg[ch_cathode][0]-raw_eeg[ch_anode][0])
    bipolar_chn_val = np.squeeze(np.array(bipolar_chn_val))

    bipolar_info = mne.create_info(
        ch_names=bipolar_chn_names,
        sfreq=raw_eeg.info['sfreq'],
        ch_types='eeg',
    )

    bipolar_raw = mne.io.RawArray(bipolar_chn_val, info=bipolar_info, verbose=False)
    return bipolar_raw