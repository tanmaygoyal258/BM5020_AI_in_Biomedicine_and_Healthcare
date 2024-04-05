import os
import scipy
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class PTB_XL_dataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(PTB_XL_dataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv , dtype = {'patient_id':str})
        df = df[df['fold'].isin(folds)] if folds is not None else df
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['AFLT','IPMI', 'IVCD', 'BIGU', 'LAO/LAE', 'ILMI', 'ISCAS', 'RAO/RAE', 'WPW', 'ILBBB', 'IRBBB', 'ANEUR', 'PSVT', 'PAC', 'ISCIL', 'INJAL', 'ISC_', 'IMI', 'PVC', 'LNGQT', 'AFIB', '2AVB', '1AVB', 'NST_', 'DIG', 'LVH', 'ISCLA', 'INJAS', 'PMI', 'ISCIN', 'SEHYP', 'NORM', 'ISCAL', 'PACE', 'STACH', 'ISCAN', 'ALMI', 'EL', '3AVB', 'INJIL', 'CLBBB', 'CRBBB', 'INJLA', 'RVH', 'LAFB', 'INJIN', 'NDT', 'LPFB', 'AMI', 'LMI', 'ASMI', 'IPLMI']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self,index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        ecg_data = scipy.io.loadmat(os.path.join(self.data_dir, f'{patient_id}.mat'))['val']
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-15000:, self.use_leads]
        result = np.zeros((15000, self.nleads)) # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return patient_id , torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)
