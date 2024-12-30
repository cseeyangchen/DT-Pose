import random
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 对3D姿态、原始WiFi信号和提取WiFi特征进行质量评估
def evaluate_clustering_quality(wifi, wifi_features, labels):
    wifi = wifi.reshape(wifi.shape[0], -1)
    # 计算轮廓系数 (Silhouette Score)
    silhouette_avg = silhouette_score(wifi_features, labels) - silhouette_score(wifi, labels)
    print(f"Silhouette Score (more than zero is better): {silhouette_avg:.4f}")
    # 计算Davies-Bouldin Index (DBI)
    dbi_score = davies_bouldin_score(wifi_features, labels) - davies_bouldin_score(wifi, labels)
    print(f"Davies-Bouldin Index (less than zero is better): {dbi_score:.4f}")
    # 计算Calinski-Harabasz Index (CHI)
    chi_score = calinski_harabasz_score(wifi_features, labels) - calinski_harabasz_score(wifi, labels)
    print(f"Calinski-Harabasz Index (more than zero is better): {chi_score:.4f}")


def uniformity_loss(features):
    # Normalize features to the unit sphere
    features = F.normalize(features, dim=1)
    # Compute pairwise cosine similarities
    similarities = torch.mm(features, features.T)  # Shape (n, n)
    # Mask out the diagonal (self-similarities)
    n = features.shape[0]
    mask = torch.eye(n, device=features.device).bool()
    similarities = similarities.masked_fill(mask, 0)
    # Calculate the uniformity loss as the average squared similarity
    unif_loss = torch.mean(similarities**2)
    return unif_loss

def infonce_loss(batch_data, temperature=0.5):
    # Get the size of the batch (2n, d)
    n, d = batch_data.shape[0] // 2, batch_data.shape[1]
    # Split batch_data into two parts: frame 1 and frame 2
    frame_1, frame_2 = batch_data[:n], batch_data[n:]
    # Compute similarity matrix between all pairs (n x n)
    # Similarity is computed using the dot product between frame_1 and frame_2
    similarity_matrix = torch.mm(frame_1, frame_2.T) / temperature
    # Extract positive logits (diagonal elements)
    positive_logits = torch.diag(similarity_matrix).view(n, 1)
    # Extract negative logits by masking out the positive pairs
    mask = torch.eye(n, device=batch_data.device).bool()
    negative_logits = similarity_matrix[~mask].view(n, -1)
    # Concatenate positive and negative logits
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    # Create labels for the positive pairs (the positive pair is always at index 0)
    labels = torch.zeros(n, dtype=torch.long, device=batch_data.device)
    # Calculate the cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_pck_pckh(dt_kpts,gt_kpts,thr,align=False,dataset='mmfi-csi'):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)

    if align == True:
        dt = dt.transpose(0,2,1)
        gt = gt.transpose(0,2,1)
        preds_hip = dt[:, 0, :]  # (batch_size, 3)
        gts_hip = gt[:, 0, :]  # (batch_size, 3)
        # Compute the offset to align the hips
        offset = gts_hip - preds_hip  # (batch_size, 3)
        # Expand the offset to all joints and apply the translation
        dt = dt + offset[:, None, :]  # Broadcasting the offset to all joints (batch_size, num_joints, 3)
        dt = dt.transpose(0,2,1)
        gt = gt.transpose(0,2,1)

    assert(dt.shape[0]==gt.shape[0])
    kpts_num=gt.shape[2] #keypoints
    ped_num=gt.shape[0] #batch_size
    #compute dist
    if dataset=='mmfi-csi':
        scale=np.sqrt(np.sum(np.square(gt[:,:,5]-gt[:,:,12]),1)) #right shoulder--left hip
    elif dataset=='person-in-wifi-3d':
        scale=np.sqrt(np.sum(np.square(gt[:,:,6]-gt[:,:,4]),1)) #right shoulder--left hip
    elif dataset=='wipose':
        scale=np.sqrt(np.sum(np.square(gt[:,:,5]-gt[:,:,8]),1)) #right shoulder--left hip
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #compute pck
    pck = np.zeros(gt.shape[2]+1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= thr)
        # compute average pck
    if dataset=='mmfi-csi':
        pck[17] = 100*np.mean(dist <= thr)
    elif dataset=='person-in-wifi-3d':
        pck[14] = 100*np.mean(dist <= thr)
    elif dataset=='wipose':
        pck[18] = 100*np.mean(dist <= thr)
    return pck


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    # import numpy as np

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c


def calulate_error(preds, gts, align=False):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    """
    N = preds.shape[0]
    num_joints = preds.shape[1]

    if align == True:
        preds_hip = preds[:, 0, :]  # (batch_size, 3)
        gts_hip = gts[:, 0, :]  # (batch_size, 3)
        # Compute the offset to align the hips
        offset = gts_hip - preds_hip  # (batch_size, 3)
        # Expand the offset to all joints and apply the translation
        preds = preds + offset[:, None, :]  # Broadcasting the offset to all joints (batch_size, num_joints, 3)

    # mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))
    mpjpe = np.sqrt(np.sum(np.square(preds - gts), axis=2))  #  b 17
    mpjpe_joints = mpjpe
    mpjpe = mpjpe.mean(1)

    pampjpe = np.zeros([N, num_joints])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    # pampjpe = np.mean(pampjpe)
    pampjpe_joints = pampjpe
    pampjpe = pampjpe.mean(1)

    return mpjpe, pampjpe, mpjpe_joints, pampjpe_joints