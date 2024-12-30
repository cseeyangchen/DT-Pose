import torch
import torch.nn as nn
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from model.utils.tgcn import ConvTemporalGraphical
from model.utils.graph import Graph
import torch.nn.functional as F
from torchvision import models, transforms
import torch.fft as fft
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=(114,10),   # 32
                 patch_size=(2,2),   # 2
                 emb_dim=256,
                 num_layer=12,  # 12
                 num_head=4,
                 input_dim=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size[0]) *(image_size[1] // patch_size[1]), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        # self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.patchify = torch.nn.Conv2d(input_dim, emb_dim, (patch_size[0], patch_size[1]),(patch_size[0], patch_size[1]))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()


    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def feature_extract(self, img):
        batch_num, anntena_num, subcarrier_num, frequency_num = img.size()
        features = []
        features_list = []
        # for sample_idx in range(batch_num):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))[:,0,:]
        return features

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=(114,10),   # 32
                 patch_size=(2,2),
                 emb_dim=256,
                 num_layer=4,    # 4
                 num_head=4,
                 output_dim=3
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size[0]) *(image_size[1] // patch_size[1]) + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2 )
        self.head = torch.nn.Linear(emb_dim, output_dim * patch_size[0] * patch_size[1])   # 3
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=image_size[0]//patch_size[0])

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=(114, 10),   # 32
                 patch_size=(2,2),
                 emb_dim=256,   # 256
                 encoder_layer=12,
                 encoder_head=4,
                 decoder_layer=4,
                 decoder_head=4,
                 input_dim=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, input_dim, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, input_dim)
        # projector
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

    def forward(self, img, flag):
        features, backward_indexes = self.encoder(img)
        if flag == "test":
            unmasked_features = self.encoder.feature_extract(img)
            predicted_img, mask = self.decoder(features,  backward_indexes)
            return predicted_img, mask, unmasked_features
        else:
            predicted_img, mask = self.decoder(features,  backward_indexes)
            cl_feature = self.predictor(features[1:,:,:].mean(0))
            return predicted_img, mask, features[0], cl_feature


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = torch.matmul(adj, x)  
        x = self.fc(x)
        return self.relu(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attn_weights = None

    def forward(self, src):
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        self.attn_weights = attn_weights  
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




class ViT_Pose_Decoder(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, keypoints=17, coor_num=3, token_num=285, dataset='mmfi-csi', num_person=1) -> None:
        super().__init__()
        self.keypoints = keypoints
        self.dataset = dataset
        self.num_person = num_person
        self.coor_num = coor_num
        # pretrain model
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.emb_dim = self.cls_token.size()[2]
        for p in self.parameters():
            p.requires_grad = False

        # Task Prompt
        self.pose_prompt = nn.Parameter(torch.zeros(self.emb_dim, self.keypoints*self.num_person))
        trunc_normal_(self.pose_prompt, std=.02)
        # GCN
        self.gconv1 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.gconv2 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.gconv3 = GraphConvLayer(self.emb_dim, self.emb_dim)
        self.adj_matrix = self.generate_adjacency_matrix(num_joints=self.keypoints, dataset=self.dataset)  
        # Transformer
        self.transformer_encoder1 = CustomTransformerEncoderLayer(d_model=self.emb_dim, nhead=4)
        self.transformer_encoder2 = CustomTransformerEncoderLayer(d_model=self.emb_dim, nhead=4)
        self.transformer_encoder3 = CustomTransformerEncoderLayer(d_model=self.emb_dim, nhead=4)
        self.fc= nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim//4),
            nn.ReLU(),
            nn.Linear(self.emb_dim//4, self.coor_num)
        )
       

    def forward(self, img):
        # b 3 114 10 -> b 297 3 114 10
        batch_num, anntena_num, subcarrier_num, frequency_num = img.size()
        features = []
        features_list = []
        # for sample_idx in range(batch_num):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = features[:,1:,:].mean(1)

        # reshape
        x = features.unsqueeze(2).expand(batch_num, self.emb_dim, self.keypoints*self.num_person)
        # task prompt
        x = x + self.pose_prompt.unsqueeze(0).expand(batch_num, -1, -1)
        if self.dataset == 'mmfi-csi' or self.dataset == 'wipose':
            x = x.permute(0, 2, 1) 
        elif self.dataset == 'person-in-wifi-3d':
            x = x.view(-1, self.emb_dim, self.keypoints).permute(0,2,1)  
       
        # GCN
        adj = self.adj_matrix.to(feature.device)
        x = self.gconv1(x, adj)
        x = self.gconv2(x, adj)
        x = self.gconv3(x, adj)
        # Transformer
        x = self.transformer_encoder1(x)
        x = self.transformer_encoder2(x)
        x = self.transformer_encoder3(x)
        # linear
        pose = self.fc(x)  
        if self.dataset == 'person-in-wifi-3d':
            pose = pose.view(batch_num, -1, self.keypoints, self.coor_num)
        
        return pose, features

    def generate_adjacency_matrix(self, num_joints=17, dataset='mmfi-csi'):
        adj_matrix = torch.zeros((num_joints, num_joints))
        if dataset == 'mmfi-csi':
            connections = [
                [0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9,10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
            ]
        elif dataset == 'person-in-wifi-3d':
            connections = [
                [0, 1], [0, 2], [2, 5],
                [3, 0], [4, 2], [5, 7],
                [6, 3], [7, 3], [8, 4], [9,5],
                [10, 6], [11, 7], [12, 9],
                [13, 11]
            ]
        elif dataset == 'wipose':
            connections = [
                [17,15],[0,15],[0,14],[14,16],
                [1,2],[2,3],[3,4],
                [1,5],[5,6],[6,7],
                [1,8],[8,9],[9,10],
                [1,11],[11,12],[12,13]
            ]
        for i, j in connections:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  
        return adj_matrix







if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(32, 3, 114, 10)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(features.shape)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
