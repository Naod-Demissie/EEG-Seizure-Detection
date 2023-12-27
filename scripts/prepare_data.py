import os
import sys
import mne
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(ROOT_PATH)

from utils.preprocessing.convert_montage import convert_to_bipolar
from utils.plots import plot_eeg_lbl, plot_eeg_tse

from utils import read_lbl_file
from utils.data_loader import return_data_segment

from utils.features.time_domain import *
from utils.features.frequency_domain import *



with open(f'{ROOT_PATH}/data/processed/seizure_files.txt', 'r+') as seiz_f:
    train_paths = [path.replace('\n', '') for path in seiz_f.readlines()]
    for idx, train_path in enumerate(train_paths):
        with open(f'{ROOT_PATH}/data/processed/extracted_features/temp/extracted_files.txt', 'a+') as extracted_f:
            with open(f'{ROOT_PATH}/data/processed/extracted_features/temp/extracted_files.txt', 'r+') as extracted_ff:
                extracted_paths = [path.replace('\n', '') for path in extracted_ff.readlines()]

                if train_path not in extracted_paths:
                    try:
                        print(f'{idx+1}. {train_path}')
                        eeg_segments, segment_labels, sfreq = return_data_segment(train_path, segment_duration=1.0, step_size=1.0)
                        file_name = train_path.split('/')[-1].split('.')[0]

                        eeg_segment_features = []
                        for eeg_segment in tqdm(eeg_segments):

                            stat_feature = StatisticalFeatures(eeg_segment)
                            amp_features = AmplitudeFeatures(eeg_segment)
                            hjorth_param = HjorthParameters(eeg_segment)
                            entr_features = EntropyFeatures(eeg_segment)

                            spectral_features = SpectralFeatures(eeg_segment, sample_rate=sfreq)

                            time_domain_features = [
                                stat_feature.mean(), stat_feature.std_deviation(), stat_feature.variance(),
                                stat_feature.mode(), stat_feature.median(), stat_feature.skewness(),
                                stat_feature.kurtosis(), stat_feature.minimum(), stat_feature.maximum(),
                                stat_feature.coefficient_of_variation(), *stat_feature.quartiles(), stat_feature.inter_quartile_range(),

                                amp_features.energy(), amp_features.average_power(), amp_features.root_mean_squared(),
                                amp_features.line_length(), amp_features.zero_crossings(), amp_features.local_extrema(),

                                hjorth_param.activity(), hjorth_param.mobility(), hjorth_param.complexity(),

                                entr_features.shannon_entropy(),
                                # entr_features.approximate_entropy(m=2, r=0.2),
                                # entr_features.sample_entropy(m=2, r=0.5) , entr_features.permutation_entropy(m=3),
                                # entr_features.weighted_permutation_entropy(m=3), entr_features.fuzzy_entropy(m=2, r=0.5),
                                # entr_features.distribution_entropy(num_bins=10), entr_features.svd_entropy(num_components=5)
                            ]

                            frequency_features = [
                                spectral_features.mean(), spectral_features.variance(), spectral_features.skewness(),
                                spectral_features.kurt(), spectral_features.energy(),
                                *spectral_features.frequency_bands(band_ranges=[(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]),
                                spectral_features.peak_frequency(), spectral_features.frequency_centroid(),
                                spectral_features.bandwidth(), spectral_features.spectral_entropy(),
                                *spectral_features.cross_frequency_ratios(band_ranges=[(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)])
                            ]

                            eeg_segment_features.append(time_domain_features + frequency_features)
                        eeg_segment_features = np.array(eeg_segment_features)
                        np.savez_compressed(f'{ROOT_PATH}/data/processed/extracted_features/win_size_1s/{file_name}_feat', eeg_segment_features)
                        np.savez_compressed(f'{ROOT_PATH}/data/processed/extracted_features/win_size_1s/{file_name}_lbl', segment_labels)

                        print(eeg_segment_features.shape)
                        extracted_f.write(f'{train_path}\n')
                    except:
                        with open(f'{ROOT_PATH}/data/processed/extracted_features/temp/error_files.txt', 'a+') as err_f:
                            err_f.write(f'{train_path}\n')
