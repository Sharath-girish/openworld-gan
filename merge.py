import os
import sys
import copy
import time
import torch
import utils
import config
import shutil
import numpy as np
from config import Config
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity

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

# Obtain clusterwise means for set of feature vectors
def get_means(X, labels):
    means = []
    for i in range(len(np.unique(labels))):
        means.append(np.mean(X[labels==i], axis=0))
    return means

# Obtain k-nearest neighbour indices in X for each feature vector in X
def knn_gpu(X,k, device):
    n = X.shape[0]
    batchsize = 1024
    n_batches = n//1024 +(1 if n%1024!=0 else 0)
    knn = np.zeros((n,k))
    X = torch.Tensor(X).to(device)
    for i in range(n_batches):
        dists = get_pairwise_distances(X[i*batchsize:(i+1)*batchsize],X)
        _, indices = torch.topk(dists, k+1, dim=1, largest=False, sorted=True)
        knn[i*batchsize:(i+1)*batchsize] = indices[:,1:].detach().cpu().numpy()
    return knn

# Performs merging of clusters defined with features {X_in} and corresponding labels {labels_in}
def merge_clusters_nn(X_in, labels_in, connection_type, device):
    num_clusters = np.unique(labels_in).shape[0]
    num_samples = X_in.shape[0]
    centres = np.zeros((num_clusters,X_in.shape[1]))
    labels = np.zeros((num_clusters,num_samples))
    labels[labels_in.astype(np.int64),np.arange(num_samples)] = 1
    labels = labels/np.sum(labels,axis=1,keepdims=True)
    centres = np.dot(labels,X_in) # Obtain cluster centers
    # Find nearest neighbor to each cluster center
    knn_indices = knn_gpu(centres, 1, device)[:,0].astype(np.int64) 
    adj_mat = np.zeros((num_clusters,num_clusters))
    # Create graph with cluster centres as nodes and edges set to 1 for nearest neighbor pairs
    adj_mat[np.arange(num_clusters),knn_indices[np.arange(num_clusters)]] = 1
    graph = csr_matrix(adj_mat)
    # Obtain connected components from graph with connected components forming merged clusters
    n_components, label_mapping = connected_components(csgraph=graph, directed=True, 
                                            connection= connection_type, return_labels=True)
    # Return label mapping based on which clusters belonging to which connected component
    return label_mapping


def get_hash(X, perm_indices, K, device):
    N = perm_indices.size(0)
    Xhash = torch.zeros(X.size(0),N).to(device)
    for i in range(N):
        Xnew = X[:,perm_indices[i]][:,:K]
        Xhash[:,i] = torch.argmax(Xnew,dim=1)
    Xhash = F.one_hot(Xhash.reshape(-1).long(),num_classes=K).reshape(X.size(0),N,K).float()
    return Xhash.reshape(X.size(0),N*K)

def run_merge(step, conf, save_conf):
    # Obtain conf and OOD, clustering, merge save paths for current iteration
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    conf_merge = config.get_conf_merge(conf_step)
    ood_save_path = utils.get_ood_path(root_path,step)
    clustering_save_path = utils.get_clustering_path(root_path,step)
    merge_save_path = utils.get_merge_path(root_path,step)

    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
    num_seen_classes = len(train_classes)

    # Obtain evaluation set features and corresponding cluster labels
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    features = torch.load(os.path.join(clustering_save_path,'features.pt'))
    X_feats_eval = features['X_feats_eval']
    cluster_mapping = features['cluster_mapping']
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    num_old_clusters = features['num_old_clusters']

    # Transform feature using WTA hash if enabled for computing cluster distances
    if conf_merge['use_wta']:
        conf_ood = config.get_conf_ood(conf_step)
        N, K = conf_ood['num_hashes'], conf_ood['window_size']
        perm_indices = torch.load(os.path.join(ood_save_path,'features.pt'))['perm_indices']
        X_feats_eval = get_hash(torch.Tensor(X_feats_eval).to(device),\
                                perm_indices, K, device).detach().cpu().numpy()
    
    # If not first step, first perform merge only among the new clusters from previous clustering 
    # stage to deal with overclustering
    X_in = X_feats_eval[cluster_labels>=num_old_clusters]
    if X_in.shape[0]>0:
        labels_in = cluster_labels[cluster_labels>=num_old_clusters]-num_old_clusters
        if np.unique(labels_in).shape[0]>2: # Only perform merge with more than 2 clusters
            label_mapping = merge_clusters_nn(X_in, labels_in, conf_merge['connection_type'], device)
            cluster_labels[cluster_labels>=num_old_clusters] = label_mapping[labels_in]+num_old_clusters
            utils.check_continuous(cluster_labels[cluster_labels>=num_seen_classes]-num_seen_classes)

    # Perform merge on all discovered clusters which do not belong to the seen classes
    X_in = X_feats_eval[cluster_labels>=num_seen_classes]
    if X_in.shape[0]>0:
        labels_in = cluster_labels[cluster_labels>=num_seen_classes]-num_seen_classes
        if np.unique(labels_in).shape[0]>2: # Only perform merge with more than 2 clusters
            label_mapping = merge_clusters_nn(X_in, labels_in, conf_merge['connection_type'], device)
            cluster_labels[cluster_labels>=num_seen_classes] = label_mapping[labels_in]+num_seen_classes
            utils.check_continuous(cluster_labels[cluster_labels>=num_seen_classes]-num_seen_classes)

    # Update cluster labels and save features, cluster labels of the evaluation set for subsequent stages
    cluster_mapping = [(path,new_label) for ((path,old_label),new_label) in \
                        zip(cluster_mapping,cluster_labels)]
    

    # Evaluate cluster stats using gt labels for eval set
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    yeval = features['Y_eval']
    utils.disc_stats(cluster_labels, yeval)
    print('Cluster stats for discovered set:')
    utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1], 
                        save_path=os.path.join(merge_save_path,'cluster_stats_disc.png'))
    print('Cluster stats for full evaluation set:')
    cluster_labels = utils.get_attributed_labels(cluster_labels,features["X_feats_eval"],device)
    utils.cluster_stats(cluster_labels, yeval, 
                        save_path=os.path.join(merge_save_path,'cluster_stats.png'))

    torch.save({
                'cluster_mapping':cluster_mapping, 
                'X_feats_eval': features['X_feats_eval'], 
                'Y_eval': features['Y_eval']
                }, 
                os.path.join(merge_save_path,'features.pt')
            )

    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(merge_save_path,"config.yaml"))