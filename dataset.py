from typing import List, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from copy import deepcopy
from h5py import File
import random
from tqdm import tqdm
import torch.utils.data as tdata


def load_dict_from_csv(file, cols, sep="\t"):
    if isinstance(file, str):
        df = pd.read_csv(file, sep=sep)
    elif isinstance(file, pd.DataFrame):
        df = file
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


class InferenceDataset(tdata.Dataset):
    def __init__(self,
                 audio_file):
        super(InferenceDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aids = list(self.aid_to_h5.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        return aid, feat


class TrainDataset(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx,
                 augment = 'no'):
        super(TrainDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx
        self.augment = augment
    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        if self.augment  == 'noi':
            #对每一帧加高斯噪声
            for i in range(feat.size(0)):
                feat[i] += random.gauss(0,0.01)
        elif self.augment  == 'mix':
            #交换音频最开始的一部分和最后一部分
            rand_num = random.randint(2,6)
            time_step = feat.size(0)
            seg_length = int(time_step/rand_num)
            temp_1 = deepcopy(feat[0:seg_length])
            temp_2 = deepcopy(feat[-seg_length:])
            feat[0:seg_length] = temp_2
            feat[-seg_length:] = temp_1
        elif self.augment  == 'pitch':
            #所有帧的频率整体向上或向下平移
            rand_num = random.randint(2,10)
            for i in range(feat.size(0)):
                #rand_num为奇数时频率整体升高
                if rand_num%2 == 0:
                    feat[i,rand_num:] = deepcopy(feat[i,0:(feat.size(1)-rand_num)])
                    feat[i,0:rand_num] = 0
                #rand_num为偶数时频率整体下降
                else:
                    feat[i,0:(feat.size(1)-rand_num)] = deepcopy(feat[i,rand_num:] )
                    feat[i,-rand_num] = 0
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        return aid, feat, target


def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    if isinstance(tensorlist[0], np.ndarray):
        tensorlist = [torch.as_tensor(arr) for arr in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    length = [tensor.shape[0] for tensor in tensorlist]
    return padded_seq, length


def sequential_collate(return_length=True, length_idxs: List=[]):
    def wrapper(batches):
        seqs = []
        lens = []
        for idx, data_seq in enumerate(zip(*batches)):
            if isinstance(data_seq[0],
                          (torch.Tensor, np.ndarray)):  # is tensor, then pad
                data_seq, data_len = pad(data_seq)
                if idx in length_idxs:
                    lens.append(data_len)
            else:
                data_seq = np.array(data_seq)
            seqs.append(data_seq)
        if return_length:
            seqs.extend(lens)
        return seqs
    return wrapper
