import os
import utils
import torch
import config
import shutil
import argparse
import numpy as np
from stages import *
from config import Config

# List of stages to be executed in the corresponding order
stage_names = ['classifier','generate_features', 'ood_detection', 'clustering', 
               'merge', 'refine']
stage_classes = [StageClassifier, StageGenerateFeatures, StageOOD, StageClustering, 
                 StageMerge, StageRefine]


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None,
    help='config filepath in yaml format, can be list separated by ;')
parser.add_argument('--filepath', type=str, default=None,
    help='filepath containing the features and labels to evaluate')

def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path,exist_ok=False)

def main(conf, filepath):
    if filepath:
        features = torch.load(filepath,map_location='cpu')
        cluster_mapping = features['cluster_mapping']
        yeval = features['Y_eval']
        print_str = f'            Evaluating metrics for file {filepath}          '
        print('-'*len(print_str))
        print(print_str)
        print('-'*len(print_str))
        cluster_labels = np.array([item[1] for item in cluster_mapping])
        num_clusters = np.unique(cluster_labels[cluster_labels!=-1]).shape[0]
        num_samples = cluster_labels[cluster_labels!=-1].shape[0]
        utils.disc_stats(cluster_labels, yeval)
        print('Cluster stats for discovered set:')
        utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1])
        print('Cluster stats for full evaluation set:')
        network_feats = features["X_feats_eval"]
        labels = np.zeros((num_clusters,num_samples))
        labels[cluster_labels[cluster_labels!=-1].astype(np.int64),np.arange(num_samples)] = 1
        labels = labels/np.sum(labels,axis=1,keepdims=True)
        centroids = torch.matmul(torch.Tensor(labels),torch.Tensor(network_feats[cluster_labels!=-1]))
        cluster_labels[cluster_labels==-1] = torch.argmin(utils.get_pairwise_distances( \
                                                torch.Tensor(network_feats[cluster_labels==-1]),\
                                                centroids,"cosine"),dim=1).detach().cpu().numpy()

        utils.cluster_stats(cluster_labels, yeval)
    else:
        conf_common = config.get_conf_common(conf)
        classes_list = Config(config_filepath='class_list.yaml')
        num_steps = conf_common['num_iters']
        final_stage = conf_common['final_stage']
        assert num_steps>=1
        assert final_stage in stage_names

        root_path = conf_common["save_path"]
        for step in range(1,num_steps+1):
            step_save_path = os.path.join(root_path,f'step{step}') # Save path for current iteration
            num_stages = stage_names.index(final_stage)+1 if step==num_steps else len(stage_classes)
            conf_classifier = config.get_conf_classifier(config.get_conf_step(conf,step=step))
            conf_gen_features = config.get_conf_gen_features(config.get_conf_step(conf,step=step))
            for stage in range(num_stages):
                if stage<=1:
                    continue # Ignore classifier and feature gen stage for evaluation
                stage_name = stage_names[stage]
                stage_save_path = os.path.join(step_save_path,stage_name) # Save path for current stage
                filepath = os.path.join(stage_save_path,'features.pt') # Path to cluster labels
                features = torch.load(filepath, map_location='cpu')

                print_str = f'              Iteration {step}: stage {stage_name}                '
                print('-'*len(print_str))
                print(print_str)
                print('-'*len(print_str))

                cluster_mapping = features['cluster_mapping']
                yeval = features['Y_eval']
                cluster_labels = np.array([item[1] for item in cluster_mapping])
                num_clusters = np.unique(cluster_labels[cluster_labels!=-1]).shape[0]
                num_samples = cluster_labels[cluster_labels!=-1].shape[0]
                train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
                eval_classes = sorted(classes_list[conf_gen_features['eval_classes']]['eval_classes'])

                utils.att_stats(cluster_labels, yeval, train_classes, eval_classes)
                utils.disc_stats(cluster_labels, yeval)
                print('Cluster stats for discovered set:')
                utils.cluster_stats(cluster_labels[cluster_labels!=-1], yeval[cluster_labels!=-1])
                print('Cluster stats for full evaluation set:')

                # Set non-labeled samples to nearest neighbour cluster before evaluation
                network_feats = features["X_feats_eval"]
                labels = np.zeros((num_clusters,num_samples))
                labels[cluster_labels[cluster_labels!=-1].astype(np.int64),np.arange(num_samples)] = 1
                labels = labels/np.sum(labels,axis=1,keepdims=True)
                centroids = torch.matmul(torch.Tensor(labels),torch.Tensor(network_feats[cluster_labels!=-1]))
                cluster_labels[cluster_labels==-1] = torch.argmin(utils.get_pairwise_distances( \
                                                        torch.Tensor(network_feats[cluster_labels==-1]),\
                                                        centroids,"cosine"),dim=1).detach().cpu().numpy()

                utils.cluster_stats(cluster_labels, yeval)
            

# Create config dictionary from yaml file to keep track of hyperparameters across stages and iterations
if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    conf = {}
    if args.config:
        conf = Config(use_args=True)
    main(conf, args.filepath)
