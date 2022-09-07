import os
import sys
import time
import utils
import torch
import config
import numpy as np
from cv2 import normalize
from config import Config
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Obtains N different permutations for an {n-feats}-dimensional feature space for the WTA hash
def get_hashperm(N, n_feats, device):
    indices = torch.zeros(N,n_feats).to(device)
    for i in range(N):
        indices[i,:] = torch.randperm(n_feats).to(device)
    return indices

# Obtain the hashed transformation for a set of feature vectors {X} based on the 
# permutation indices {perm_indices} and the window size {K} for the WTA hash
# The resulting transformed feature vector has a feature dimension of N*K
def get_hash(X, perm_indices, K):
    N = perm_indices.size(0)
    Xhash = torch.zeros(X.size(0),N).to(X.device)
    for i in range(N):
        Xnew = X[:,perm_indices[i]][:,:K]
        Xhash[:,i] = torch.argmax(Xnew,dim=1)
    Xhash = F.one_hot(Xhash.reshape(-1).long(),num_classes=K).reshape(X.size(0),N,K).float()
    return Xhash.reshape(X.size(0),N*K)

# Obtain the {frac}th quantile for a vector x, frac lies in (0-1)
def quantiles(x, frac):
    length = x.size(0)
    return torch.topk(x,max(1,int(length*(1-frac))))[0][-1]

# Returns OOD mask of whether evaluation features {eval_feats} are in-distribution/OOD
# with respect to the distribution of the train features {train_feats} based on a 
# threshold. {perm_indices} contain the permutation indices for the WTA hash
def ood_wta(conf_ood, train_feats, eval_feats, perm_indices, device):
    ntrain, n_feats = train_feats.size()
    neval = eval_feats.size(0)
    N, K = conf_ood['num_hashes'], conf_ood['window_size']
    # Run in batched mode to fit features in GPU memory, reduce if necessary
    train_batchsize = 10000
    n_batches = ntrain//train_batchsize+(1 if ntrain%train_batchsize!=0 else 0)

    # Obtain the hashed feature vectors for the train set
    train_feats_hash = torch.zeros(ntrain,N*K)
    for i in range(n_batches):
        train_feats_hash[i*train_batchsize:(i+1)*train_batchsize] = \
        get_hash(train_feats[i*train_batchsize:(i+1)*train_batchsize].to(device), 
                 perm_indices, K).detach().cpu()

    # Obtain the pairwise distances of train features
    dist = N - torch.matmul(train_feats_hash, train_feats_hash.t())
    # Set the threshold based on the quantile or the mean-std values of the 
    # vector of train distances. Lower threshold leads to tighter bounds on classifying
    # an evaluation feature vector as in-distribution
    if conf_ood['thresh_type'] == 'mean_std':
        threshold = torch.mean(dist).item() + \
                            conf_ood['thresh_value']*torch.std(dist).item()
    elif conf_ood['thresh_type'] == 'quantiles':
        threshold = quantiles(dist.reshape(-1).to(device).half(),\
                            conf_ood['thresh_value']).item()
    else:
        raise Exception('Unknown threshold type {}'.format(conf_ood['thresh_type']))

    # Obtain the hashed feature vectors for the evaluation set
    eval_batchsize = 10000
    n_batches = neval//eval_batchsize+(1 if neval%eval_batchsize!=0 else 0)
    eval_feats_hash = torch.zeros(neval,N*K).to(device)
    for i in range(n_batches):
        eval_feats_hash[i*eval_batchsize:(i+1)*eval_batchsize] = \
        get_hash(eval_feats[i*eval_batchsize:(i+1)*eval_batchsize].to(device), 
                 perm_indices, K)

    # Compute the average distances for each vector in eval set from all vectors in train set
    # If the average distance is below the threshold, it is classified as in-distribution
    eval_batchsize = 1000 # Run in batched mode to fit in GPU memory
    n_batches = ntrain//eval_batchsize + (1 if ntrain%eval_batchsize!=0 else 0)
    dist = torch.zeros(neval).to(device)
    for j in range(n_batches):
        dist += torch.sum(N-torch.matmul(\
                train_feats_hash[j*eval_batchsize:(j+1)*eval_batchsize].to(device),
                eval_feats_hash.t()),dim=0).to(device)
    dist = dist/ntrain # Average the summed distance over the train set
    # Return OOD classification mask, sets to 1 (True) if in-distribution else 0
    return dist < threshold
    

# Main function for obtaining OOD masks for each cluster and performing 
# attribution for images which are in-distribution to each cluster
def generate_ood_masks(step, conf, save_conf):
    # Obtain conf and feature generation and OOD save paths for current iteration
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    conf_ood = config.get_conf_ood(conf_step)
    features_save_path = utils.get_gen_features_path(root_path,step)
    ood_save_path = utils.get_ood_path(root_path,step)

    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
    num_seen_classes = len(train_classes)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    # Obtain the features and the labels for the train and evaluation set from the previous
    # feature generation stage
    features = torch.load(os.path.join(features_save_path,'features.pt'))
    Xtrain, Xeval, ytrain, yeval = features['X_feats_train'], features['X_feats_eval'],\
                                   features['Y_train'], features['Y_eval']
    # Obtain predictions for the evaluation set
    Xpreds_eval = torch.Tensor(features['X_preds_eval']).argmax(dim=1).to(device)
    cluster_mapping = features['cluster_mapping']
    # Cluster labels for eval set
    cluster_labels = torch.Tensor([item[1] for item in cluster_mapping]).to(device)
    num_classes = int(cluster_labels.max().item()+1)
    # Check that the cluster labels in the discovered set are continuous
    if num_classes>num_seen_classes:
        utils.check_continuous(cluster_labels[cluster_labels>=num_seen_classes]-num_seen_classes)

    # Normalize the features based on the train distribution statistics 
    # Helps with OOD detection using WTA hash
    normalize_type = conf_ood['normalize_type']
    if normalize_type != 'none':
        if normalize_type == 'mean_std':
            scaler = StandardScaler() 
        elif normalize_type == 'min_max':
            scaler = MinMaxScaler()
        else:
            raise Exception('Undefined normalization type ', normalize_type)
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xeval = scaler.transform(Xeval)

    Xtrain = torch.Tensor(Xtrain)
    Xeval = torch.Tensor(Xeval)
    ytrain = torch.Tensor(ytrain).long()
    yeval = torch.Tensor(yeval).long()

    # Obtain permutation indices for the hashing transform
    _, n_feats = Xtrain.size()
    N = conf_ood['num_hashes']
    perm_indices = get_hashperm(N, n_feats, device).long()

    # Obtain OOD masks of evaluation set w.r.t all seen classes in the training set
    # 1 means in-distribution and 0 means OOD
    ood_seen_masks = torch.zeros(Xeval.size(0),num_seen_classes).to(device)
    for tr in range(num_seen_classes):
        sys.stdout.write(f'\rGenerating OOD masks for seen class {tr+1}/{num_seen_classes}')
        Xtrainclass = Xtrain[ytrain==tr].to(device)
        ood_mask = ood_wta(conf_ood, Xtrainclass, Xeval, perm_indices, device)
        ood_seen_masks[:,tr] = ood_mask

    # Any image is OOD with respect to training set if it is OOD to all seen classes
    ood_seen_mask = (torch.sum(ood_seen_masks,dim=1)!=0).float()

    # If there are additional discovered clusters (if not the first iteration)
    # Perform additional round of OOD detection of evaluation set set w.r.t discovered clusters
    if num_classes>num_seen_classes:
        ood_unseen_masks = torch.zeros(Xeval.size(0),num_classes-num_seen_classes).to(device)
        for c in range(num_seen_classes,num_classes):
            sys.stdout.write(f'\rGenerating OOD masks for cluster {c-num_seen_classes+1}'+\
                             f'/{num_classes-num_seen_classes}      ')
            Xtrainclass = Xeval[cluster_labels==c].to(device)
            ood_mask = ood_wta(conf_ood, Xtrainclass, Xeval, perm_indices, device)
            ood_unseen_masks[:,c-num_seen_classes] = ood_mask
        # Image is OOD to discovered set if it is OOD to all discovered clusters
        ood_unseen_mask = (torch.sum(ood_unseen_masks,dim=1)!=0).float()
    else:
        ood_unseen_mask = torch.zeros(Xeval.size(0)).to(device)

    sys.stdout.write('\rGenerated OOD masks'+' '*50+'\n')
    # Attribute only labels belonging to non-discovered set (cluster_labels is -1)
    # Attribute only if network predicts seen class (label<{num_seen_classes})
    seen_update_mask = torch.logical_and(
                        torch.logical_and(ood_seen_mask==1,cluster_labels==-1),
                        Xpreds_eval<num_seen_classes)
    # Attribute based on network prediction only if network predicts unseen class
    unseen_update_mask = torch.logical_and(
                        torch.logical_and(ood_unseen_mask==1,cluster_labels==-1),
                        Xpreds_eval>=num_seen_classes)
    # Perform network attribution based on network prediction for non-discovered set
    cluster_labels[seen_update_mask] = Xpreds_eval[seen_update_mask].float()
    cluster_labels[unseen_update_mask] = Xpreds_eval[unseen_update_mask].float()

    if num_classes>num_seen_classes:
        utils.check_continuous(cluster_labels[cluster_labels>=num_seen_classes]-num_seen_classes)

    # Update cluster labeling for newly attributed images
    cluster_mapping = [(path,new_label) for ((path,old_label),new_label) in \
                        zip(cluster_mapping,cluster_labels.detach().cpu().numpy())]

    # Evaluate cluster stats using gt labels for eval set
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    yeval = np.array(features['Y_eval'])
    utils.disc_stats(cluster_labels, yeval)
    print('Cluster stats for discovered set:')
    utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1], 
                        save_path=os.path.join(ood_save_path,'cluster_stats_disc.png'))
    print('Cluster stats for full evaluation set:')
    cluster_labels = utils.get_attributed_labels(cluster_labels,features["X_feats_eval"],device)
    utils.cluster_stats(cluster_labels, yeval, 
                        save_path=os.path.join(ood_save_path,'cluster_stats.png'))

    # Save features and labels of the evaluation set for subsequent stages
    torch.save({
                'perm_indices':perm_indices, 
                'X_feats_train':features['X_feats_train'],
                'X_feats_eval': features['X_feats_eval'], 
                'Y_eval': features['Y_eval'],
                'cluster_mapping':cluster_mapping
                }, 
                os.path.join(ood_save_path,'features.pt')
            )

    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(ood_save_path,"config.yaml"))
