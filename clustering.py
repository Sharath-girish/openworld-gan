import os
import sys
import time
import utils
import torch
import config
import numpy as np
from config import Config
import torch.functional as F 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Obtain pairwise distances between 2 sets of feature vectors based on distance metric
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

# k-means++ for initializing k-means clusters for faster convergence
def kmeans_init(data, k, device):
    ns, feats = data.size()
    centroids = torch.zeros(k, feats)
    idx = torch.randint(ns,(1,))
    centroids[0,:] = data[idx,:]
    dists = torch.zeros(k-1,ns)
    for c in range(1,k):
        sys.stdout.write('\r'+'Initializing cluster {}/{}'.format(c+1,k))
        batch_size = 20000
        X = centroids[c-1:c,:].to(device)
        n_batches = int(ns/batch_size)+(1 if ns%batch_size!=0 else 0)
        cur_dists = torch.zeros(1,ns).to(device)
        for i in range(n_batches):
            Y = data[i*batch_size:(i+1)*batch_size,:].to(device)
            cur_dists[:,i*batch_size:(i+1)*batch_size] = get_pairwise_distances(X,Y)
        if c>1:
            min_dists = torch.min(cur_dists.squeeze(),min_dists)
        else:
            min_dists = cur_dists.squeeze()
        probs = min_dists/torch.sum(min_dists)
        idx = np.random.choice(np.arange(ns), p=probs.detach().cpu().numpy())
        centroids[c,:] = data[idx,:]
    sys.stdout.write('\r'+' '*50)
    sys.stdout.write('\r'+ '')
    return centroids

# k-means algorithm with gpu capability for {k} clusters and {max_iter} iterations
# Initializes centroids if not provided using the k-means++ algorithm
def kmeans_gpu(data, k, device, max_iter = 100, centroids = None):
    if centroids is None:
        centroids = kmeans_init(data, k, device)
    else:
        assert centroids.size(0) == k and centroids.size(1) == data.size(1)
    init_centroids = centroids
    Ns = data.size(0)
    labels = torch.zeros(Ns)
    c_list = [centroids.detach().cpu().numpy()]
    for it in range(max_iter):
        sys.stdout.write('\r'+'Iteration {}/{}'.format(it+1,max_iter))
        dists = torch.zeros(k,Ns)
        batch_size = 20000
        X = centroids.to(device)
        n_batches = int(Ns/batch_size)+(0 if Ns%batch_size==0 else 1)
        new_centroids = torch.zeros_like(centroids).to(device)
        new_counts = torch.zeros(k).to(device)
        labels = torch.zeros(Ns)
        dist_vals = torch.zeros(Ns)
        dummy_centroids = torch.zeros_like(new_centroids).to(device)
        for i in range(n_batches):
            Y = data[i*batch_size:(i+1)*batch_size,:].to(device)
            dists = get_pairwise_distances(X,Y)
            # print(dists.size())
            vals, indices = torch.min(dists, dim=0)
            dist_vals[i*batch_size:(i+1)*batch_size] = vals
            val_indices, inverse_indices, count_indices = torch.unique(indices, return_inverse = True, return_counts=True, sorted = False)
            new_centroids[val_indices] = new_centroids[val_indices] + dummy_centroids.index_put((indices,), Y, accumulate=True)[val_indices]
            new_counts[val_indices] += count_indices
            labels[i*batch_size:(i+1)*batch_size] = indices.detach().cpu()

        centroids = new_centroids/new_counts.unsqueeze(1)
        c_list.append(centroids.detach().cpu().numpy())
    sys.stdout.write('\r'+' '*50)
    sys.stdout.write('\r'+ '')

    return labels.long(), centroids.detach().cpu()

def get_hash(X, perm_indices, K, device):
    N = perm_indices.size(0)
    Xhash = torch.zeros(X.size(0),N).to(device)
    for i in range(N):
        Xnew = X[:,perm_indices[i]][:,:K]
        Xhash[:,i] = torch.argmax(Xnew,dim=1)
    Xhash = F.one_hot(Xhash.reshape(-1).long(),num_classes=K).reshape(X.size(0),N,K).float()
    return Xhash.reshape(X.size(0),N*K)

# Main function for running k-means clustering on the non-discovered set
def run_kmeans(step, conf, save_conf):
    # Obtain conf and OOD and clustering save paths for current iteration
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    conf_clustering = config.get_conf_cluster(conf_step)
    ood_save_path = utils.get_ood_path(root_path,step)
    clustering_save_path = os.path.join(root_path,f"step{step}","clustering")

    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
    num_seen_classes = len(train_classes)

    # Obtain the features and the labels for the train and evaluation set
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    features = torch.load(os.path.join(ood_save_path,'features.pt'))
    Xtrain, Xeval = features['X_feats_train'], features['X_feats_eval']
    cluster_mapping = features['cluster_mapping']
    cluster_labels = np.array([item[1] for item in cluster_mapping])

    # Set of images/features to be clustered (non-discovered set)
    Xcluster = Xeval[cluster_labels==-1]
    num_classes = cluster_labels.max()+1

    if Xcluster.shape[0] > conf_clustering['num_clusters']:
        # Normalize the features to be clustered based on the train distribution statistics 
        if conf_clustering['normalize_type'] == 'mean_std':
            scaler = StandardScaler()
            scaler.fit(Xtrain)
            Xcluster = scaler.transform(Xcluster)
            Xcluster = torch.Tensor(Xcluster)
        elif conf_clustering['normalize_type'] == 'min_max':
            scaler = MinMaxScaler()
            scaler.fit(Xtrain)
            Xcluster = scaler.transform(Xcluster)
            Xcluster = torch.Tensor(Xcluster)
        elif conf_clustering['normalize_type'] == 'none':
            Xcluster = torch.Tensor(Xcluster)

        # Perform k-means clustering on the non-discovered set
        cluster_labels_new, _ = kmeans_gpu(Xcluster, conf_clustering['num_clusters'], 
                                        device, max_iter = conf_clustering['max_iter'])

        # Bad initialization of k-means sometimes leads to 1 or 0 clusters, rerun the algorithm 
        while torch.unique(cluster_labels_new).size(0)<=1:
            print('Rerunning k-means')
            cluster_labels_new, _ = kmeans_gpu(Xcluster, conf_clustering['num_clusters'], 
                                            device, max_iter = conf_clustering['max_iter'])

        utils.check_continuous(cluster_labels_new)

        # Add the newly added clusters to the cluster labels
        cluster_labels[cluster_labels==-1] = cluster_labels_new.detach().cpu().numpy()\
                                            +num_classes

        # Remove clusters below a size threshold
        # Cluster numbers are reordered to maintain continuous numbering of clusters
        cluster_update = {}
        counter = 0
        for cl in np.unique(cluster_labels):
            if cl==num_classes:
                num_old_clusters = counter
            if cluster_labels[cluster_labels==cl].shape[0]>=conf_clustering['size_threshold']:
                cluster_update[cl] = counter
                counter += 1
            else:
                cluster_update[cl] = -1
        
        updated_cluster_labels = np.array([cluster_update[cl] for cl in cluster_labels])
        utils.check_continuous(updated_cluster_labels[updated_cluster_labels>=num_seen_classes]\
                                -num_seen_classes)

        cluster_mapping = [(path,new_label) for ((path,old_label),new_label) in \
                            zip(cluster_mapping,updated_cluster_labels)]
    else:
        # Do not perform clustering if number of images less than clusters
        num_old_clusters = num_classes
        cluster_mapping = features['cluster_mapping']

    # Evaluate cluster stats using gt labels for eval set
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    yeval = features['Y_eval']
    utils.disc_stats(cluster_labels, yeval)
    print('Cluster stats for discovered set:')
    utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1], 
                        save_path=os.path.join(clustering_save_path,'cluster_stats_disc.png'))
    print('Cluster stats for full evaluation set:')
    cluster_labels = utils.get_attributed_labels(cluster_labels,features["X_feats_eval"],device)
    utils.cluster_stats(cluster_labels, yeval, 
                        save_path=os.path.join(clustering_save_path,'cluster_stats.png'))

    # Save features and cluster labels of the evaluation set for subsequent stages
    torch.save({
                'num_old_clusters':num_old_clusters, 
                'X_feats_eval': features['X_feats_eval'], 
                'Y_eval': features['Y_eval'], 
                'cluster_mapping':cluster_mapping
                },
                os.path.join(clustering_save_path,'features.pt')
            )
    
    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(clustering_save_path,"config.yaml"))