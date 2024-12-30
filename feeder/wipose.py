import os
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
import h5py
import pywt
import mat73


class WiPose(Dataset):
    def __init__(self, split, data_root):
        self.action_dict = {'bend':0, 'circle':1, 'crouch':2, 'jump':3, 'wave':4, 'walk':5, 'throw':6, 'standup':7, 'sitdown':8, 'run':9, 'push':10, 'pull':11}
        self.split = split
        self.data_root = data_root
        if self.split == 'training':
            self.filename_list_path = os.path.join(self.data_root, 'Train_data_list.txt')
            self.data_root = os.path.join(self.data_root, 'Train_Amplitude_DWT')
            self.data_list = self.load_train_data(self.filename_list_path)
            print(f'Traning sample {len(self.data_list)}.')
        elif self.split == 'validation':
            self.filename_list_path = os.path.join(self.data_root, 'Test_data_list.txt')
            self.data_root = os.path.join(self.data_root, 'Test_Amplitude_DWT')
            self.data_list = self.load_test_data(self.filename_list_path)
            print(f'Test sample {len(self.data_list)}.')
        else:
            print('No Training and Testing Settings!')


    def load_train_data(self, file_path):
        file_name_list = []
        # flag = 0
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                filename = lines.split()[0]
                file_name_list.append(filename)
                # flag += 1
                # if flag == 1000:
                #     break
        data_info = []
        for filename in file_name_list:
            data_path = os.path.join(self.data_root, (str(filename)+'.npz'))
            frame_idx = int(filename.split('-frame')[1])
            next_frame_file_name = filename.split('-frame')[0]+'-frame'+str(frame_idx+1).zfill(3)
            action_label = self.action_dict[filename.split('_')[0]]
            if next_frame_file_name in file_name_list:
                data_dict = {   
                                'frame_idx': frame_idx,
                                'data_path': data_path,
                                'label': action_label
                                }
                data_dict['data_path_next_frame'] = os.path.join(self.data_root, (str(next_frame_file_name)+'.npz'))
                data_info.append(data_dict)
        return data_info

    def load_test_data(self, file_path):
        data_info = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                filename = lines.split()[0]
                data_path = os.path.join(self.data_root, (str(filename)+'.npz'))
                frame_idx = int(filename.split('-frame')[1])
                action_label = self.action_dict[filename.split('_')[0]]
                data_dict = {   
                                'frame_idx': frame_idx,
                                'data_path': data_path,
                                'label': action_label
                                }
                data_info.append(data_dict)
        return data_info

    def read_frame(self, csi_path):
        data = np.load(csi_path)
        csi_data = data['CSI']
        csi_data = (csi_data - np.min(csi_data)) / (np.max(csi_data) - np.min(csi_data))
        pose_data = data['SkeletonPoints'].transpose(1,0)*0.001   # 18, 2
        return csi_data, pose_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        csi_data, pose_data =  self.read_frame(item['data_path'])
        sample = {
                    'csi':csi_data,
                    'pose':pose_data,
                    'label':item['label'],
                    'split':self.split
                    }
        if self.split == 'training':
            csi_data_next_frame, pose_data_next_frame = self.read_frame(item['data_path_next_frame'])
            sample['csi_next_frame'] = csi_data_next_frame
            sample['pose_next_frame'] = pose_data_next_frame
        return sample

 
def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    batch_data = {'split': batch[0]['split']}

    _output = [np.array(sample['pose']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output
    for split in batch_data['split']:
        if split == 'training':
            # next frame
            _output_next_frame = [np.array(sample['pose_next_frame']) for sample in batch]
            _output_next_frame = torch.FloatTensor(np.array(_output_next_frame))
            batch_data['output_next_frame'] = _output_next_frame
    
    _input = [np.array(sample['csi']) for sample in batch]
    _input = torch.FloatTensor(np.array(_input))
    batch_data['input_wifi-csi'] = _input
    # for split in batch_data['split']:
    if batch_data['split'] == 'training':
        # next frame
        _input_next_frame = [np.array(sample['csi_next_frame']) for sample in batch]
        _input_next_frame = torch.FloatTensor(np.array(_input_next_frame))
        batch_data['input_wifi-csi_next_frame'] = _input_next_frame

    batch_data['label'] = torch.tensor([sample['label'] for sample in batch])
            

    return batch_data

def wp_make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        num_workers=8
    )
    return loader
