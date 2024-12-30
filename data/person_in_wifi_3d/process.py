import h5py
import pywt
import torch
import numpy as np
import os
import glob


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  
            if not lines:
                break
            file_name_list.append(lines.split()[0])
    return file_name_list

def phase_deno(csi):
    #input csi shape (3*3*30*20)
    ph_rx1 = CSI_sanitization(csi[0,:,:,:])
    ph_rx2 = CSI_sanitization(csi[1,:,:,:])
    ph_rx3 = CSI_sanitization(csi[2,:,:,:])
    csi_phde = np.concatenate((np.expand_dims(ph_rx1,axis=0), 
                                np.expand_dims(ph_rx2,axis=0), 
                                np.expand_dims(ph_rx3,axis=0),))
    return csi_phde


def CSI_sanitization(csi_rx):
    one_csi = csi_rx[0,:,:]
    two_csi = csi_rx[1,:,:]
    three_csi = csi_rx[2,:,:]
    pi = np.pi
    M = 3  # 天线数量3
    N = 30  # 子载波数目30
    T = one_csi.shape[1]  # 总包数
    fi = 312.5 * 2  # 子载波间隔312.5 * 2
    csi_phase = np.zeros((M, N, T))
    for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
        csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
        csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
        csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
        ai = np.tile(2 * pi * fi * np.array(range(N)), M)
        bi = np.ones(M * N)
        ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
        A = np.dot(ai, ai)
        B = np.dot(ai, bi)
        C = np.dot(bi, bi)
        D = np.dot(ai, ci)
        E = np.dot(bi, ci)
        rho_opt = (B * E - C * D) / (A * C - B ** 2)
        beta_opt = (B * D - A * E) / (A * C - B ** 2)
        temp = np.tile(np.array(range(N)), M).reshape(M, N)
        csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
    antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
    antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
    antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
    antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0), 
                                    np.expand_dims(antennaPair_Two,axis=0), 
                                    np.expand_dims(antennaPair_Three,axis=0),))
    return antennaPair


def dwt_amp(csi):
    w = pywt.Wavelet('dB11')
    list = pywt.wavedec(abs(csi), w,'sym')
    csi_amp = pywt.waverec(list, w)
    return csi_amp

def main():
    train_list_path = 'train_data/train_data_list.txt'
    test_list_path = 'test_data/test_data_list.txt'
    for setting_path in [train_list_path, test_list_path]:
        name_list = load_file_name_list(setting_path)
        for name in name_list:
            csi_path = os.path.join(setting_path.split('/')[0],'csi',(str(name)+'.mat'))
            keypoint_path = os.path.join(setting_path.split('/')[0],'keypoint',(str(name)+'.npy'))
            # deal
            csi = h5py.File(csi_path)['csi_out']
            csi = csi['real'] + csi['imag']*1j
            csi = np.array(csi).transpose(3,2,1,0)
            csi = csi.astype(np.complex128)
            csi_amp = dwt_amp(csi).transpose(1,0,2,3).reshape((3,90,20))  # phase
            csi_ph = phase_deno(csi)
            csi_ph = np.angle(csi_ph).transpose(1,0,2,3).reshape((3,90,20))
            csi = np.concatenate((csi_amp, csi_ph), axis=1)
            # store
            csi_amp_path = os.path.join(setting_path.split('/')[0], 'csi_amplitude')
            if not os.path.exists(csi_amp_path):
                os.makedirs(csi_amp_path)
            np.save(f'{csi_amp_path}/{str(name)}.npy', csi_amp)

            csi_ph_path = os.path.join(setting_path.split('/')[0], 'csi_phase')
            if not os.path.exists(csi_ph_path):
                os.makedirs(csi_ph_path)
            np.save(f'{csi_ph_path}/{str(name)}.npy', csi_ph)

            # csi_ap_path = os.path.join(setting_path.split('/')[0], 'csi_ap'):
            # if not os.path.exists(csi_ap_path):
            #     os.makedirs(csi_ap_path)
            # np.save(f'{csi_ap_path}/{str(name)}.npy', csi)
            
            print(f'Down: {csi_path}')
    print('All Down.')

            


if __name__ == "__main__":
    main()