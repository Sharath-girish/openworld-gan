from typing import Union
import torch.backends.cudnn as cudnn

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

def setup_cuda(seed:Union[float, int]=None, local_rank:int=0):
    if seed is not None:
        seed = int(seed) + local_rank
        # setup cuda
        cudnn.enabled = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    #torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True # set to false if deterministic
    torch.set_printoptions(precision=10)
    # cudnn.deterministic = True
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def cuda_device_names()->str:
    return ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# Check that labels have continuous unique values from 0 to maximum value
def check_continuous(label):
    if torch.is_tensor(label):
        if label.size(0)==0:
            return
        max_val = label.max()
        uq_labels = torch.unique(label[label!=-1])
        assert uq_labels.min().item()==0 and uq_labels.size(0) == (max_val.item()+1)
    else:
        if label.shape[0]==0:
            return
        max_val = label.max()
        uq_labels = np.unique(label[label!=-1])
        assert uq_labels.min() == 0 and uq_labels.shape[0] == (max_val+1)

def get_classifier_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","classifier")

def get_gen_features_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","generate_features")

def get_ood_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","ood_detection")

def get_clustering_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","clustering")

def get_merge_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","merge")

def get_refine_path(root_path, step:int):
    return os.path.join(root_path,f"step{step}","refine")

def get_pairwise_distances(X,Y, metric='euclidean'):
    dot_pdt = torch.mm(X,Y.t())
    if metric=='euclidean': 
        square_norm_1 = torch.sum(X**2, dim = 1).unsqueeze(1)
        square_norm_2 = torch.sum(Y**2, dim = 1).unsqueeze(1)
        return torch.sqrt(torch.abs(square_norm_1 + square_norm_2.t() - 2.0*dot_pdt))
    elif metric=='cosine':
        Xnorm = X/torch.norm(X, p=2, dim=1).unsqueeze(1)
        Ynorm = Y/torch.norm(Y, p=2, dim=1).unsqueeze(1)
        return torch.abs(1.0 - torch.mm(Xnorm, Ynorm.t()))

def get_attributed_labels(cluster_labels, network_feats, device):
    num_clusters = np.unique(cluster_labels[cluster_labels!=-1]).shape[0]
    num_samples = cluster_labels[cluster_labels!=-1].shape[0]
    labels = np.zeros((num_clusters,num_samples))
    labels[cluster_labels[cluster_labels!=-1].astype(np.int64),np.arange(num_samples)] = 1
    labels = labels/np.sum(labels,axis=1,keepdims=True)
    centroids = torch.matmul(torch.Tensor(labels).to(device),\
                             torch.Tensor(network_feats[cluster_labels!=-1]).to(device))
    cluster_labels[cluster_labels==-1] = torch.argmin(get_pairwise_distances( \
                                        torch.Tensor(network_feats[cluster_labels==-1]).to(device),\
                                        centroids,"cosine"),dim=1).detach().cpu().numpy()  
    return cluster_labels

def cluster_stats(predicted, targets, save_path=None):
    n_clusters = np.unique(predicted).size
    n_classes  = np.unique(targets).size
    num = np.zeros([n_clusters,n_classes])
    unique_targets = np.unique(targets)
    for i,p in enumerate(np.unique(predicted)):
        class_labels = targets[predicted==p]
        num[i,:] = np.sum(class_labels[:,np.newaxis]==unique_targets[np.newaxis,:],axis=0)
    sum_clusters = np.sum(num,axis=1)
    purity = np.max(num,axis=1)/(sum_clusters+(sum_clusters==0).astype(sum_clusters.dtype))
    indices = np.argsort(-purity)

    if save_path is not None:
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(purity[indices],color='red')
        ax1.set_xlabel('Cluster index')
        ax1.set_ylabel('Purity')
        ax2 = ax1.twinx()
        ax2.plot(sum_clusters[indices])
        ax2.set_ylabel('Cluster size')
        plt.legend(('Purity','Cluster size'))
        plt.show()
        plt.title('Cluster size and purity of discovered clusters')
        plt.savefig(save_path)
    print('Data points {} Clusters {}'.format(np.sum(sum_clusters).astype(np.int64), n_clusters))
    print('Average purity: {:.4f} '.format(np.sum(purity*sum_clusters)/np.sum(sum_clusters))+\
          'NMI: {:.4f} '.format(normalized_mutual_info_score(targets, predicted))+\
          'ARI: {:.4f} '.format(adjusted_rand_score(targets, predicted)))

def disc_stats(predicted, targets):
    n_clusters = np.unique(predicted).size
    n_classes  = np.unique(targets).size
    n = np.zeros([n_clusters,n_classes])
    unique_targets = np.unique(targets)
    for i,p in enumerate(np.unique(predicted)):
        class_labels = targets[predicted==p]
        n[i,:] = np.sum(class_labels[:,np.newaxis]==unique_targets[np.newaxis,:],axis=0)
    cluster_assignments = n.argmax(axis=1)
    n_class_discovered = np.unique(cluster_assignments).shape[0]
    frac_samples_discovered = predicted[predicted!=-1].shape[0]/predicted.shape[0]
    print(f'Classes discovered: {n_class_discovered}/{n_classes}, '+\
          f'% samples discovered: {frac_samples_discovered*100:.2f}')

def att_stats(predicted, targets, train_classes, eval_classes):
    seen_unseen_mapping = np.array([1 if c in sorted(train_classes) else 0 \
                            for i,c in enumerate(sorted(eval_classes))])
    y_preds = np.array([1 if c<len(train_classes) and c>=0 else 0 for c in predicted])
    y_gt = seen_unseen_mapping[targets.astype(np.int64)]
    print(f'Attribution accuracy: {np.mean(y_preds==y_gt)*100:.2f}%')