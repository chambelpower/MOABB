import mne
from scipy.io import loadmat
from moabb.datasets.base import BaseDataset
import numpy as np

class BCIAUTP300(BaseDataset):
    """Dataset for the P300-based BCI study with ASD individuals."""

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            sessions_per_subject=7,
            events={"Target": 2, "NonTarget": 1},
            code="BCIAUTP300",
            interval=[-0.2, 1.2],  # From -200 ms to 1200 ms
            paradigm="p300",
            doi="",
        )

    def _get_single_subject_data(self, subject):
        """Load and process the EEG data for a given subject."""
        subject = f'{subject:02d}' if subject < 10 else str(subject)
        base_path = f'C:\\Users\\diver\\benchmarks\\benchmarks\\MOABB\\data\\SBJ{subject}'
        sessions = {}

        for session_num in range(1, 8):
            session_str = f'S{session_num:02d}'
            session_path = f'{base_path}\\{session_str}\\'

            # Load data from files
            train_data = loadmat(f'{session_path}Train\\trainData.mat')['trainData']
            train_events = np.loadtxt(f'{session_path}Train\\trainEvents.txt', dtype=int)
            train_targets = np.loadtxt(f'{session_path}Train\\trainTargets.txt', dtype=int)
            # Load test data and events
            test_data = loadmat(f'{session_path}Test\\testData.mat')['testData']
            test_events = np.loadtxt(f'{session_path}Test\\testEvents.txt', dtype=int)
            test_targets = np.loadtxt(f'{session_path}Test\\testTargets.txt', dtype=int)

            # Concatenate train and test data along the time axis
            combined_data = np.concatenate((train_data, test_data), axis=2)  # [channels x epoch x (train + test)]

            # Combine train and test events/targets
            combined_events = np.concatenate((train_events, test_events))
            combined_targets = np.concatenate((train_targets, test_targets))

            # Channel names, types, and sampling frequency
            ch_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
            ch_types = ['eeg'] * 8
            sfreq = 250  # Sampling frequency in Hz

            # Create MNE info structure
            info = mne.create_info(ch_names, sfreq, ch_types)

            # Set a standard montage (e.g., 'standard_1020')
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)

            # Reshape combined data from [channels x epoch x event] to [channels x (epoch * event)]
            combined_data_reshaped = np.reshape(combined_data, (combined_data.shape[0], -1))

            # Create MNE RawArray
            combined_raw = mne.io.RawArray(combined_data_reshaped, info)

            # Create events array [sample index, 0, event type] with mapped targets
            if len(combined_events) == len(combined_targets):
                events = np.column_stack((combined_events, np.zeros_like(combined_events), combined_targets))
            else:
                raise ValueError("Mismatch between combined events and combined targets lengths")

            # Create annotations from the events array
            annotations = mne.Annotations(onset=events[:, 0] / sfreq,
                                      duration=0,  # Typically, event markers have a 0-duration
                                      description=[str(e) for e in events[:, 2]])

            # Set annotations to the Raw object
            combined_raw.set_annotations(annotations)

            #print("Annotations:", combined_raw.annotations)

            # Store session data
            sessions[str(session_num-1)] = {"0": combined_raw}

        return sessions


    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return the path to the subject's data."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return []  # Update this if you need to manage actual download paths
