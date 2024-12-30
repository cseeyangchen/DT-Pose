import h5py
import pywt
import torch
import numpy as np
import os
import glob
import mat73
from collections import defaultdict


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split()[0])
    return file_name_list


def dwt_amp(csi):
    w = pywt.Wavelet('dB11')
    list = pywt.wavedec(abs(csi), w,'sym')
    csi_amp = pywt.waverec(list, w)
    return csi_amp

def generate_name_list():
    for folder_dir in ['Test', 'Train']:
        with open(f'{folder_dir}_data_list.txt', 'w') as f:
            for filename in os.listdir(folder_dir):
                f.write(filename.split('.')[0]+'\n')


# 读取并解析数据
def parse_data(file_path):
    data = defaultdict(lambda: defaultdict(list))  # {action: {person: [frames]}}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 拆分数据，例如 bend_029-frame043
            action, person_frame = line.split("_")
            person, frame = person_frame.split("-frame")
            # 将帧号转换为整数并存储
            data[action][person].append(int(frame))
    return data

# 统计动作、人数和帧数
def analyze_data(data):
    results = []
    for action, persons in sorted(data.items()):
        results.append(f"Action: {action}")
        results.append(f"  Number of people: {len(persons)}")
        for person, frames in sorted(persons.items()):
            frames.sort()  # 确保帧号有序
            results.append(f"    Person {person}: {len(frames)} frames ({frames[0]} to {frames[-1]})")
    return results


def store_analysis():
    train_list_path = 'Train_data_list.txt'
    test_list_path = 'Test_data_list.txt'
    for setting_path in [train_list_path, test_list_path]:
        data = parse_data(setting_path)
        results = analyze_data(data)
        with open(setting_path.split('_')[0]+'_Parse.txt', "w") as f:
            for line in results:
                f.write(line + "\n")

def main():
    train_list_path = 'Train_data_list.txt'
    test_list_path = 'Test_data_list.txt'
    for setting_path in [train_list_path, test_list_path]:
        name_list = load_file_name_list(setting_path)
        for name in name_list:
            csi_path = os.path.join(setting_path.split('_')[0],(str(name)+'.mat'))
            # deal
            data_mat = mat73.loadmat(csi_path)
            csi = data_mat['CSI']
            csi_amp = np.array(csi).transpose(3,2,1,0).reshape((3,90,5))   # 3 3 30 5
            # csi = np.array(csi).transpose(2,3,1,0)  # 3 3 30 5
            # csi_amp = dwt_amp(csi).transpose(1,0,2,3).reshape((3,90,5))
            keypoints = np.array(data_mat['SkeletonPoints']).reshape((3, 18))[:2,:]
            # store
            csi_amp_path = setting_path.split('_')[0]+'_Amplitude_DWT'
            if not os.path.exists(csi_amp_path):
                os.makedirs(csi_amp_path)
            np.savez(f'{csi_amp_path}/{str(name)}.npz', CSI=csi_amp, SkeletonPoints=keypoints)
            print(f'Down: {csi_path}')
    print('All Down.')

            


if __name__ == "__main__":
    # generate_name_list()
    # store_analysis()
    main()
    