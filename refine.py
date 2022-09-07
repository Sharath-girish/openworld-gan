import os
import sys
import time
import torch
import utils
import config
import pickle
import numpy as np
from config import Config
import torch.nn.functional as F
import matplotlib.pyplot as plt
from thundersvm import SVC, OneClassSVM
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity



def run_refine(step, conf, save_conf):
    # Obtain conf and merge, refine save paths for current iteration
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    conf_refine = config.get_conf_refine(conf_step)
    merge_save_path = utils.get_merge_path(root_path,step)
    refine_save_path = utils.get_refine_path(root_path,step)

    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
    num_seen_classes = len(train_classes)

    # Obtain evaluation set features and corresponding cluster labels
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    features = torch.load(os.path.join(merge_save_path,'features.pt'))
    X_feats_eval = features['X_feats_eval']
    cluster_mapping = features['cluster_mapping']
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    num_samples = X_feats_eval.shape[0]
    num_clusters = int(cluster_labels.max()+1)

    # Create SVMs for each cluster which is not part of seen classes
    # and maintain predictions of each SVM on the eval set
    predictions = np.zeros((num_clusters,num_samples)).astype(np.int64)
    kernel_type = conf_refine['kernel_type']
    pred_path = os.path.join(refine_save_path,f"predictions_s1_{kernel_type}.pickle")
    if not os.path.exists(pred_path):
        for c in np.unique(cluster_labels[cluster_labels>=num_seen_classes].astype(np.int64)):
            sys.stdout.write('\rFitting model {}/{}'.format(c+1,num_clusters))
            model = SVC(kernel=kernel_type) # SVM for current cluster
            # Positive labels are features belonging to the cluster while 
            # negative labels are everything else in the discovered set
            y = (cluster_labels[cluster_labels!=-1]==c).astype(np.int64)
            model.fit(X_feats_eval[cluster_labels!=-1],y) # Fit SVM
            # Obtain predictions of the SVM for the full evaluation set
            predictions[c] = model.predict(X_feats_eval)
        with open(pred_path,"wb") as f:
            pickle.dump(predictions,f)
        sys.stdout.write('\rSVM models fit'+' '*20+'\n')
    else:
        with open(pred_path,"rb") as f:
            predictions = pickle.load(f)
    
    counter = num_seen_classes
    cluster_update = {c: c for c in np.arange(-1,num_seen_classes)}
    for c in np.unique(cluster_labels[cluster_labels>=num_seen_classes].astype(np.int64)):
        # Using SVM predictions as a proxy for cluster purity
        # If purity below a threshold or number of positive predictions below a threshold, 
        # discard the cluster

        if np.sum(predictions[c,cluster_labels==c]==1)\
            /np.sum(cluster_labels==c) <= conf_refine['purity_threshold'] or \
            np.sum(predictions[c,cluster_labels==c]==1) <= conf_refine['size_threshold']:
            cluster_update[c] = -1
        else:
            # Discard samples in the current cluster with negative SVM predictions
            cluster_labels[np.logical_and(predictions[c]==0,cluster_labels==c)] = -1
            cluster_update[c] = counter
            counter += 1

    # Update cluster labels 
    updated_cluster_labels = np.array([cluster_update[cl] for cl in cluster_labels])
    utils.check_continuous(updated_cluster_labels[updated_cluster_labels>=num_seen_classes]\
                                -num_seen_classes)
    cluster_mapping = [(path,new_label) for ((path,old_label),new_label) in \
                        zip(cluster_mapping,updated_cluster_labels)]

    
    # Evaluate cluster stats using gt labels for eval set
    cluster_labels = np.array([item[1] for item in cluster_mapping])
    yeval = features['Y_eval']

    utils.disc_stats(cluster_labels, yeval)
    print('Cluster stats for discovered set:')
    utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1], 
                        save_path=os.path.join(refine_save_path,'cluster_stats_disc.png'))
    print('Cluster stats for full evaluation set:')
    cluster_labels = utils.get_attributed_labels(cluster_labels,features["X_feats_eval"],device)
    utils.cluster_stats(cluster_labels, yeval, 
                        save_path=os.path.join(refine_save_path,'cluster_stats.png'))


    # Save features, cluster labels of the evaluation set for subsequent stages
    torch.save({
                'cluster_mapping':cluster_mapping,
                'X_feats_eval':features['X_feats_eval'], 
                'Y_eval': features['Y_eval']
                },
                os.path.join(refine_save_path,'features.pt')
            )

    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(refine_save_path,"config.yaml"))
