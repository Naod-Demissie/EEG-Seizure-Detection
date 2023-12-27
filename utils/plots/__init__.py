import os
import sys
import mne
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt


sys.path.append('/content/drive/MyDrive/EEG-Seizure-Detection')
from utils import read_lbl_file
from utils.preprocessing.convert_montage import convert_to_bipolar


def plot_eeg_tse(eeg_path, to_bipolar=True, plot_height=2, plot_width=190, save_dst=None):
    if to_bipolar:
        eeg_data = convert_to_bipolar(eeg_path)
    else:
        eeg_data = mne.io.read_raw_edf(eeg_path, verbose=False)
    labels = eeg_data.ch_names
    num_channels = len(labels)
    sfreq = eeg_data.info['sfreq']

    lbl_path = eeg_path.replace('.edf', '.tse_bi')
    tse_label_df = df = pd.read_csv(lbl_path, sep=' ', header=0, names=['start time(s)', 'stop time(s)', 'class code', 'probability'])

    figsize = (plot_width, plot_height * num_channels)
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=figsize)
    plt.subplots_adjust(hspace=0.5)

    for i in range(num_channels):
        axes[i].plot(eeg_data.get_data()[i], label=labels[i])
        axes[i].set_yticks([])
        axes[i].set_ylabel(labels[i], rotation=0, ha='right')

        # Add horizontal lines at y=0
        axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Add vertical lines at every second
        for sec in range(1, len(eeg_data.get_data()[i])):
            if sec % sfreq == 0:
                axes[i].axvline(x=sec, color='gray', linestyle='--', linewidth=0.5)

        # Highlight seizure region
        for _, row in tse_label_df[tse_label_df['class code']=='seiz'].iterrows():
            start_sample = row['start time(s)'] * sfreq
            stop_sample = row['stop time(s)'] * sfreq
            axes[i].axvspan(start_sample, stop_sample, color='red', alpha=0.3, label='Seizure')
    plt.xlabel('Samples')
    plt.tight_layout()
    if save_dst:
        plt.savefig(f'{save_dst}/{eeg_path.split("/")[-1].split(".")[0]}')
    plt.show()



def plot_eeg_lbl(eeg_path, to_bipolar=True, plot_height=2, plot_width=190, save_dst=None):
    if to_bipolar:
        eeg_data = convert_to_bipolar(eeg_path)
    else:
        eeg_data = mne.io.read_raw_edf(eeg_path, verbose=False)
    labels = eeg_data.ch_names
    num_channels = len(labels)
    sfreq = eeg_data.info['sfreq']

    lbl_path = eeg_path.replace('.edf', '.lbl_bi')
    lbl_dicts = read_lbl_file(lbl_path)

    figsize = (plot_width, plot_height * num_channels)
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=figsize)
    plt.subplots_adjust(hspace=0.5)

    for i in range(num_channels):
        axes[i].plot(eeg_data.get_data()[i], label=labels[i])
        axes[i].set_yticks([])
        axes[i].set_ylabel(labels[i], rotation=0, ha='right')

        # Add horizontal lines at y=0
        axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Add vertical lines at every second
        for sec in range(1, len(eeg_data.get_data()[i])):
            if sec % sfreq == 0:
                axes[i].axvline(x=sec, color='gray', linestyle='--', linewidth=0.5)

        # Highlight seizure region
        for lbl_dict in lbl_dicts[i]:
            (start_time, stop_time), cls_code = lbl_dict
            start_idx, stop_idx = int(start_time * sfreq), int(stop_time * sfreq)
            if cls_code == 'seiz':
                axes[i].axvspan(start_idx, stop_idx, color='red', alpha=0.3)

    plt.xlabel('Samples')
    plt.tight_layout()
    if save_dst:
        plt.savefig(f'{save_dst}/{eeg_path.split("/")[-1].split(".")[0]}')
    plt.show()
