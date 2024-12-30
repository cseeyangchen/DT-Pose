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


class PersonInWif3D(Dataset):
    def __init__(self, split, data_root, experiment_name):
        self.split = split
        self.data_root = data_root
        self.experiment_name = experiment_name
        if self.split == 'training':
            self.data_root = os.path.join(self.data_root, 'train_data')
            self.filename_list_path = os.path.join(self.data_root, 'train_data_list.txt')
            self.data_list = self.load_train_data(self.filename_list_path, self.experiment_name)
            print(f'{self.experiment_name}: Traning sample {len(self.data_list)}.')
        elif self.split == 'validation':
            self.data_root = os.path.join(self.data_root, 'test_data')
            self.filename_list_path = os.path.join(self.data_root, 'test_data_list.txt')
            self.data_list = self.load_test_data(self.filename_list_path, self.experiment_name)
            print(f'{self.experiment_name}: Test sample {len(self.data_list)}.')
        else:
            print('No Training and Testing Settings!')


    def load_train_data(self, file_path, experiment_name):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                filename = lines.split()[0]
                pose_number = int(filename.split('_')[0][2])
                if experiment_name == 'one-person':
                    if pose_number == 1:
                        file_name_list.append(filename)
                elif experiment_name == 'two-person':
                    if pose_number == 2:
                        file_name_list.append(filename)
                elif experiment_name == 'three-person':
                    if pose_number == 3:
                        file_name_list.append(filename)
                elif experiment_name == 'all-person':
                    file_name_list.append(filename)
                else:
                    print('Wrong Experiment Name!')
        data_info = []
        for filename in file_name_list:
            csi_path = os.path.join(self.data_root,'csi_ap',(str(filename)+'.npy'))
            keypoint_path = os.path.join(self.data_root,'keypoint',(str(filename)+'.npy'))
            pose_number = int(filename.split('_')[0][2])
            frame_idx = int(filename.split('_')[2])
            next_frame_file_name = filename.split('_')[0]+'_'+filename.split('_')[1]+'_'+str(frame_idx+1)
            if next_frame_file_name in file_name_list:
                data_dict = {   
                                'pose_number': pose_number,
                                'frame_idx': frame_idx,
                                'csi_path': csi_path,
                                'keypoint_path':keypoint_path
                                }
                data_dict['csi_path_next_frame'] = os.path.join(self.data_root,'csi_ap',(str(next_frame_file_name)+'.npy'))
                data_dict['keypoint_path_next_frame'] = os.path.join(self.data_root,'keypoint',(str(next_frame_file_name)+'.npy'))
                data_info.append(data_dict)
        return data_info
    

    def load_test_data(self, file_path, experiment_name):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                filename = lines.split()[0]
                pose_number = int(filename.split('_')[0][2])
                if experiment_name == 'one-person':
                    if pose_number == 1:
                        file_name_list.append(filename)
                elif experiment_name == 'two-person':
                    if pose_number == 2:
                        file_name_list.append(filename)
                elif experiment_name == 'three-person':
                    if pose_number == 3:
                        file_name_list.append(filename)
                elif experiment_name == 'all-person':
                    file_name_list.append(filename)
                else:
                    print('Wrong Experiment Name!')
        data_info = []
        for filename in file_name_list:
            csi_path = os.path.join(self.data_root,'csi_ap',(str(filename)+'.npy'))
            keypoint_path = os.path.join(self.data_root,'keypoint',(str(filename)+'.npy'))
            pose_number = int(filename.split('_')[0][2])
            frame_idx = int(filename.split('_')[2])
            data_dict = {   
                            'pose_number': pose_number,
                            'frame_idx': frame_idx,
                            'csi_path': csi_path,
                            'keypoint_path':keypoint_path
                            }
            data_info.append(data_dict)
        return data_info

    def read_frame(self, csi_path):
        data = np.load(csi_path)
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        amplitude_data = data[:,:90,:]
        phase_data = data[:,90:,:]
        amplitude_data = (amplitude_data - np.min(amplitude_data)) / (np.max(amplitude_data) - np.min(amplitude_data))
        phase_data = (phase_data - np.min(phase_data)) / (np.max(phase_data) - np.min(phase_data))
        data = np.concatenate((amplitude_data, phase_data), axis=1)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        csi_data = self.read_frame(item['csi_path'])
        pose_data = np.array(np.load(item['keypoint_path']))
        sample = {
                    'csi':csi_data,
                    'pose':pose_data,
                    'split':self.split,
                    }
        if self.split == 'training':
            csi_data_next_frame = self.read_frame(item['csi_path_next_frame'])
            pose_data_next_frame = np.array(np.load(item['keypoint_path_next_frame']))
            sample['csi_next_frame'] = csi_data_next_frame
            sample['pose_next_frame'] = pose_data_next_frame
        return sample

 
def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    batch_data = {'split': batch[0]['split']}

    poses = [np.array(sample['pose']) for sample in batch]
    batch_data['person_num'] = torch.tensor([np.array(sample['pose']).shape[0] for sample in batch])
    # pad all poses to have shape (3, 14, 3)
    _output = []
    for pose in poses:
        pad_size = 3 - pose.shape[0]
        padded_pose = np.concatenate([pose, np.zeros((pad_size, 14, 3))], axis=0)
        _output.append(padded_pose)
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output
    # for split in batch_data['split']:
    #     if split == 'training':
    #         # next frame
    #         poses_next_frame = [np.array(sample['pose_next_frame']) for sample in batch]
    #         # pad all poses to have shape (3, 14, 3)
    #         _output_next_frame = []
    #         for pose in poses_next_frame:
    #             pad_size = 3 - pose.shape[0]
    #             padded_pose = np.concatenate([pose, np.zeros((pad_size, 14, 3))], axis=0)
    #             _output_next_frame.append(padded_pose)
    #         _output_next_frame = torch.FloatTensor(np.array(_output_next_frame))
    #         batch_data['output_next_frame'] = _output_next_frame
    
    _input = [np.array(sample['csi']) for sample in batch]
    _input = torch.FloatTensor(np.array(_input))
    batch_data['input_wifi-csi'] = _input
    # for split in batch_data['split']:
    if batch_data['split'] == 'training':
        # next frame
        _input_next_frame = [np.array(sample['csi_next_frame']) for sample in batch]
        _input_next_frame = torch.FloatTensor(np.array(_input_next_frame))
        batch_data['input_wifi-csi_next_frame'] = _input_next_frame
            

    return batch_data

def piw3_make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
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
