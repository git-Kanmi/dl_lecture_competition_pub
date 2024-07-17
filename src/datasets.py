import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import pywt
import scipy
from scipy import signal

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


class MyThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        def decim(data, q=12):#サンプリング数1/qにダウンサンプリング。エイリアスを考慮。
            return signal.decimate(data, q)

        def lowpass_filter(data, cutoff_frequency=30, sampling_rate=200, order=5):
            nyquist_frequency = 0.5 * sampling_rate
            normal_cutoff = cutoff_frequency / nyquist_frequency
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data

        def denoise_wavelet(signal, wavelet='db4', level=None):
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            coeffs[1:] = (pywt.threshold(coeff, value=0.5, mode='soft') for coeff in coeffs[1:])
            denoised_signal = np.array(pywt.waverec(coeffs, wavelet))
            return denoised_signal

        pre_X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        pre_X = pre_X[:,:,20:] #最初の20点は見せる前
        self.X = torch.from_numpy(decim(pre_X).copy()) #ダウンサンプリング
        #self.X = torch.from_numpy(lowpass_filter(pre_X).copy()).float() #ローパスフィルタ
        #self.X = denoise_wavelet(pre_X)[:,:,0:281] #ウェーブレット変換
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

 