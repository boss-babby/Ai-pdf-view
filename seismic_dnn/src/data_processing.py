import numpy as np
import segyio
from obspy import read
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class SeismicDataset(Dataset):
    def __init__(self, data_path, window_size=1024, stride=256, transform=None):
        """
        Args:
            data_path (str): Path to seismic data file (supports both SEG-Y and MSEED)
            window_size (int): Size of each data window
            stride (int): Stride between windows
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Load data based on file format
        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.segy' or file_ext == '.sgy':
            self.data = self._load_segy()
        else:
            self.data = read(data_path)[0].data
            
    def _load_segy(self):
        """Load SEG-Y file and convert to numpy array"""
        with segyio.open(self.data_path, 'r', strict=False) as segy:
            # Get basic information
            traces = []
            for trace in segy.trace:
                traces.append(trace)
            
            # Convert to numpy array
            data = np.array(traces).flatten()
            return data
        
        # Create windows
        self.windows = []
        for i in range(0, len(self.data) - window_size + 1, stride):
            self.windows.append(self.data[i:i + window_size])
        
        # Normalize data
        self.scaler = StandardScaler()
        self.windows = self.scaler.fit_transform(np.array(self.windows))
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        window = self.windows[idx]
        
        if self.transform:
            window = self.transform(window)
            
        return torch.FloatTensor(window)

def process_seismic_data(data, sampling_rate, filter_range=(1, 100)):
    """
    Process seismic data with filtering and normalization
    
    Args:
        data (numpy.ndarray): Raw seismic data
        sampling_rate (float): Sampling rate of the data
        filter_range (tuple): Frequency range for bandpass filter (min_freq, max_freq)
    
    Returns:
        numpy.ndarray: Processed seismic data
    """
    # Apply bandpass filter
    nyquist = sampling_rate / 2
    low = filter_range[0] / nyquist
    high = filter_range[1] / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    # Normalize
    normalized_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
    
    return normalized_data

def create_seismic_image(data, sampling_rate, window_size=256):
    """
    Create a spectrogram image from seismic data
    
    Args:
        data (numpy.ndarray): Processed seismic data
        sampling_rate (float): Sampling rate of the data
        window_size (int): Size of the window for STFT
    
    Returns:
        numpy.ndarray: Spectrogram image
    """
    frequencies, times, spectrogram = signal.spectrogram(
        data,
        fs=sampling_rate,
        window='hann',
        nperseg=window_size,
        noverlap=window_size // 2,
        detrend='constant'
    )
    
    # Convert to dB scale
    spectrogram = 10 * np.log10(spectrogram + 1e-10)
    
    return frequencies, times, spectrogram

def read_segy_metadata(segy_file):
    """
    Read metadata from a SEG-Y file
    
    Args:
        segy_file (str): Path to SEG-Y file
    
    Returns:
        dict: Dictionary containing metadata
    """
    metadata = {}
    with segyio.open(segy_file, 'r', strict=False) as segy:
        segy.mmap()
        
        # Get basic file information
        metadata['n_traces'] = len(segy.trace)
        metadata['sample_interval'] = segy.bin[segyio.BinField.Interval]
        metadata['n_samples'] = len(segy.samples)
        metadata['samples'] = list(segy.samples)
        metadata['trace_sorting'] = segy.sorting
        
        # Get binary header info
        metadata['binary_header'] = {
            'format': segy.bin[segyio.BinField.Format],
            'sample_interval': segy.bin[segyio.BinField.Interval],
            'bytes_per_sample': segy.bin[segyio.BinField.Traces]
        }
        
        # Get statistics from first trace
        try:
            first_trace = segy.trace[0]
            metadata['trace_stats'] = {
                'min': float(np.min(first_trace)),
                'max': float(np.max(first_trace)),
                'mean': float(np.mean(first_trace)),
                'std': float(np.std(first_trace))
            }
        except Exception as e:
            metadata['trace_stats'] = {'error': str(e)}
    
    return metadata

def extract_inline_slice(segy_file, inline_number):
    """
    Extract a specific inline slice from SEG-Y file
    
    Args:
        segy_file (str): Path to SEG-Y file
        inline_number (int): Inline number to extract
    
    Returns:
        numpy.ndarray: 2D array containing the inline slice
    """
    with segyio.open(segy_file, 'r', strict=False) as segy:
        try:
            data = segy.iline[inline_number]
            return data
        except (KeyError, IndexError):
            print(f"Warning: Inline {inline_number} not found in file")
            return None

def extract_crossline_slice(segy_file, crossline_number):
    """
    Extract a specific crossline slice from SEG-Y file
    
    Args:
        segy_file (str): Path to SEG-Y file
        crossline_number (int): Crossline number to extract
    
    Returns:
        numpy.ndarray: 2D array containing the crossline slice
    """
    with segyio.open(segy_file, 'r', strict=False) as segy:
        try:
            data = segy.xline[crossline_number]
            return data
        except (KeyError, IndexError):
            print(f"Warning: Crossline {crossline_number} not found in file")
            return None

def extract_time_slice(segy_file, time_index):
    """
    Extract a specific time slice from SEG-Y file
    
    Args:
        segy_file (str): Path to SEG-Y file
        time_index (int): Time index to extract
    
    Returns:
        numpy.ndarray: 2D array containing the time slice
    """
    with segyio.open(segy_file, 'r', strict=False) as segy:
        try:
            data = segy.depth_slice[time_index]
            return data
        except (KeyError, IndexError):
            print(f"Warning: Time slice {time_index} not found in file")
            return None

def prepare_training_data(data_dir, train_ratio=0.8):
    """
    Prepare training and validation datasets from SEG-Y files
    
    Args:
        data_dir (str): Directory containing SEG-Y files
        train_ratio (float): Ratio of data to use for training
    
    Returns:
        tuple: (train_files, val_files)
    """
    import os
    import random
    
    # Get all SEG-Y files
    segy_files = [f for f in os.listdir(data_dir) if f.endswith(('.sgy', '.segy'))]
    random.shuffle(segy_files)
    
    # Split into train and validation
    split_idx = int(len(segy_files) * train_ratio)
    train_files = segy_files[:split_idx]
    val_files = segy_files[split_idx:]
    
    return train_files, val_files

class SeismicDatasetV2(Dataset):
    """Enhanced SeismicDataset with better handling of SEG-Y data"""
    
    def __init__(self, data_dir, file_list, patch_size=(128, 128), stride=(64, 64), transform=None):
        """
        Args:
            data_dir (str): Directory containing SEG-Y files
            file_list (list): List of SEG-Y file names to use
            patch_size (tuple): Size of patches to extract (time_samples, traces)
            stride (tuple): Stride for patch extraction
            transform (callable, optional): Optional transform to be applied on patches
        """
        self.data_dir = data_dir
        self.file_list = file_list
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # Pre-compute patches information
        self.patches = []
        for file_name in self.file_list:
            file_path = os.path.join(data_dir, file_name)
            with segyio.open(file_path, 'r', strict=False) as segy:
                n_samples = len(segy.samples)
                n_traces = len(segy.trace)
                
                # Calculate number of patches for this file
                n_patches_v = (n_samples - patch_size[0]) // stride[0] + 1
                n_patches_h = (n_traces - patch_size[1]) // stride[1] + 1
                
                for i in range(n_patches_v):
                    for j in range(n_patches_h):
                        self.patches.append({
                            'file': file_name,
                            'start_sample': i * stride[0],
                            'start_trace': j * stride[1]
                        })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        file_path = os.path.join(self.data_dir, patch_info['file'])
        
        with segyio.open(file_path, 'r', strict=False) as segy:
            # Extract patch
            patch_data = np.zeros(self.patch_size)
            traces = segy.trace[patch_info['start_trace']:patch_info['start_trace'] + self.patch_size[1]]
            
            for i, trace in enumerate(traces):
                patch_data[:, i] = trace[patch_info['start_sample']:
                                       patch_info['start_sample'] + self.patch_size[0]]
            
            # Normalize patch
            patch_data = (patch_data - np.mean(patch_data)) / (np.std(patch_data) + 1e-6)
            
            if self.transform:
                patch_data = self.transform(patch_data)
            
            return torch.FloatTensor(patch_data).unsqueeze(0)  # Add channel dimension

def get_data_loaders(data_dir, batch_size=32, patch_size=(128, 128), stride=(64, 64)):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Directory containing SEG-Y files
        batch_size (int): Batch size for training
        patch_size (tuple): Size of patches to extract
        stride (tuple): Stride for patch extraction
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split data into train and validation
    train_files, val_files = prepare_training_data(data_dir)
    
    # Create datasets
    train_dataset = SeismicDatasetV2(data_dir, train_files, patch_size, stride)
    val_dataset = SeismicDatasetV2(data_dir, val_files, patch_size, stride)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
