import os
import argparse
import math

import yaml
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize

from feeder.mmfi import make_dataset, make_dataloader
from feeder.person_in_wifi_3d import PersonInWif3D, piw3_make_dataloader
from feeder.wipose import WiPose, wp_make_dataloader

from model.model import *
from utils import *
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training Stage")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default='config/mmfi/pretrain_config.yaml')
    args = parser.parse_args()

    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    setup_seed(config['seed'])

    batch_size = config['batch_size']
    load_batch_size = min(config['max_device_batch_size'], batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    
    # load dataset
    if config['dataset_name'] == 'mmfi-csi':
        train_dataset, val_dataset = make_dataset(config['training_semi'], config['dataset_root'], config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)   # **config['train_loader']
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=config['validation_loader']['batch_size'])   # **config['validation_loader']
    elif config['dataset_name'] == 'wipose':
        train_dataset = WiPose('training', config['dataset_root'])
        val_dataset = WiPose('validation', config['dataset_root'])
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = wp_make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = wp_make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=config['validation_loader']['batch_size'])
        pass 
    elif config['dataset_name'] == 'person-in-wifi-3d':
        train_dataset = PersonInWif3D('training', config['dataset_root'], config['experiment_name'])
        val_dataset = PersonInWif3D('validation', config['dataset_root'], config['experiment_name'])
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = piw3_make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = piw3_make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=config['validation_loader']['batch_size'])
    else:
        print('No dataset!')

    # TODO: Settings, e.g., your model, optimizer, device, ...
    writer = SummaryWriter(os.path.join('logs', config['dataset_name'], config['experiment_name'], 'pretrain', config['model_name']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['dataset_name'] == 'mmfi-csi':
        model = MAE_ViT(image_size=(114, 10),
                    patch_size=(2,2),   # (2,2)
                    encoder_layer=4,
                    encoder_head=4,
                    decoder_layer=2,
                    decoder_head=4,
                    emb_dim=256,
                    mask_ratio=config['mask_ratio']).to(device)
    elif config['dataset_name'] == 'person-in-wifi-3d':
        model = MAE_ViT(image_size=(180, 20),
                    patch_size=(2,2),
                    encoder_layer=4,
                    encoder_head=4,
                    decoder_layer=2,
                    decoder_head=4,
                    emb_dim=256,
                    input_dim=3,
                    mask_ratio=config['mask_ratio']).to(device)
    elif config['dataset_name'] == 'wipose':
        model = MAE_ViT(image_size=(90, 5),
                    patch_size=(2,1),
                    encoder_layer=4,
                    encoder_head=4,
                    decoder_layer=2,
                    decoder_head=4,
                    emb_dim=256,
                    input_dim=3,
                    mask_ratio=config['mask_ratio']).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config['base_learning_rate'] * config['batch_size'] / 256, betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    lr_func = lambda epoch: min((epoch + 1) / (config['warmup_epoch'] + 1e-8), 0.5 * (math.cos(epoch / config['total_epoch'] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)


    # TODO: Codes for training (and saving models)
    step_count = 0
    optim.zero_grad()
    for epoch in range(config['total_epoch']):
        model.train()
        losses = []
        losses_mse = []
        losses_unif = []
        losses_contrastive = []
        for batch_idx, batch_data in enumerate(train_loader):
            step_count += 1
            # Please check the data structure here.
            csi_data_current_frame = batch_data['input_wifi-csi'].to(device)
            csi_data_next_frame = batch_data['input_wifi-csi_next_frame'].to(device)   # next frame
            n, _, _, _ = csi_data_current_frame.size()
            csi_data = torch.cat((csi_data_current_frame, csi_data_next_frame), dim=0)
            # model
            predicted_csi, mask, features, cl_feature = model(csi_data, "train")
            loss_mse = torch.mean((predicted_csi[:n,:,:,:] - csi_data_current_frame) ** 2 * mask[:n,:,:,:]) / config['mask_ratio']
            loss_unif = uniformity_loss(features[:n,:])
            loss_contrastive = infonce_loss(cl_feature, temperature=0.5)
            # Calculate the current value of lambda_param based on the schedule
            cl_lambda = min(0.0001 + (0.01 - 0.0001) * (epoch / 400), 0.01)
            loss = loss_mse + 0.01 * loss_unif + cl_lambda * loss_contrastive
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            losses_mse.append(loss_mse.item())
            losses_unif.append(0.01 * loss_unif.item())
            losses_contrastive.append(cl_lambda *loss_contrastive.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        avg_loss_mse = sum(losses_mse) / len(losses_mse)
        avg_loss_unif = sum(losses_unif) / len(losses_unif)
        avg_loss_contrastive = sum(losses_contrastive) / len(losses_contrastive)
        writer.add_scalars('losses', {
                                        'train_loss': avg_loss,
                                        'train_mse_loss': avg_loss_mse,
                                        'train_unif_loss': avg_loss_unif,
                                        'train_contrastive_loss': avg_loss_contrastive,
                                    }, global_step=epoch)
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
        print(f'In epoch {epoch}, average traning loss is {avg_loss}.')

        # # TODO: Codes for test (if)
        ''' visualize the first 16 predicted images on val dataset'''
        if (epoch+1) % 50 == 0:
            model.eval()
            csi_mask_list = []
            predicted_val_csi_list = []
            val_csi_list = []
            with torch.no_grad():
                losses = []
                for batch_idx, batch_data in enumerate(val_loader):
                    val_csi = batch_data['input_wifi-csi'].to(device)
                    predicted_val_csi, mask, unmasked_features = model(val_csi, "test")
                    predicted_val_csi = predicted_val_csi * mask + val_csi * (1 - mask)
                    loss = torch.mean((predicted_val_csi - val_csi) ** 2 * mask) / config['mask_ratio']
                    losses.append(loss.item())
                    if config['dataset_name'] == 'mmfi-csi' and config['experiment_name'] == 'protocol3-s1':
                        csi_mask = val_csi * (1 - mask)  # n 3 144 10
                        csi_mask_list.append(csi_mask.data.cpu().numpy())
                        predicted_val_csi_list.append(predicted_val_csi.data.cpu().numpy())
                        val_csi_list.append(val_csi.data.cpu().numpy())
                avg_loss = sum(losses) / len(losses)
                writer.add_scalar('val_loss', avg_loss, global_step=epoch)
                if config['dataset_name'] == 'mmfi-csi' and config['experiment_name'] == 'protocol3-s1':
                    csi_mask_list = np.concatenate(csi_mask_list, axis=0)
                    predicted_val_csi_list = np.concatenate(predicted_val_csi_list, axis=0)
                    val_csi_list = np.concatenate(val_csi_list, axis=0)
                    if not os.path.exists('features/mmfi-csi/protocol3-s1/pretrain'):
                        os.makedirs('features/mmfi-csi/protocol3-s1/pretrain')
                    np.savez(os.path.join('features/mmfi-csi/protocol3-s1/pretrain', 'p3s1-test-csi.npz'), csi_mask=csi_mask_list, predicted_val_csi=predicted_val_csi_list, val_csi=val_csi_list)
            print(f'In epoch {epoch}, average val loss is {avg_loss}.')



        ''' save model '''
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['experiment_name'])
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        torch.save(model, '{}/pretrain_{}.pt'.format(weights_path, config['model_name']))


