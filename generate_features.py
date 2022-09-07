import os
import sys
import time
import utils
import torch
import config
import os.path
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data
from config import Config
import torch.nn.functional as F
import torch.utils.data as data
from networks import get_network
from datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler


def generate_features(step, conf, save_conf):
    # Get config and save paths for feature gen stage and classsifier stage
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    conf_gen_features = config.get_conf_gen_features(conf_step)
    classifier_save_path = utils.get_classifier_path(root_path, step)
    features_save_path = utils.get_gen_features_path(root_path, step)

    # Obtain train and eval classes for corresponding train and eval set of images
    # eval_classes is used only to obtain the set of images for evaluation. 
    # gt labels are used only for evaluating clustering metrics like average purity, nmi
    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])
    eval_classes = sorted(classes_list[conf_gen_features['eval_classes']]['eval_classes'])

    # If first iteration, discovered set is empty
    if step == 1:
        discovered_clusters = {}
        num_classes = len(train_classes)
    else:
        refine_save_path = utils.get_refine_path(root_path, step-1)
        features = torch.load(os.path.join(refine_save_path,"features.pt"))
        discovered_clusters = features['cluster_mapping']
        # Obtain number of clusters from cluster labels from refine stage of previous iter
        num_classes = np.max(np.array([item[1] for item in discovered_clusters]))+1
        # Create dictionary of path to eval images as keys and their cluster labels as values
        discovered_clusters = {k:v for k,v in discovered_clusters}


    # Create transform for train and eval dataset and their dataloaders
    # Train dataset contains only seen classes with gt labels while eval dataset can consist
    # of both seen and unseen classes without gt labels
    transform =  transforms.Compose([
                                    transforms.Resize(conf_classifier['resize']),
                                    transforms.CenterCrop(conf_classifier['resize']),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                    ])

    train_data_path = conf_common['train_data_path']
    eval_data_path = conf_common['eval_data_path']
    train_dataset = ImageFolder(train_data_path, train_classes, transform=transform, 
                            cache_path=os.path.join(classifier_save_path,'train_dataset_cache.pt'))
    eval_dataset = ImageFolder(eval_data_path, eval_classes, transform=transform, 
                            cache_path=os.path.join(features_save_path,'eval_dataset_cache.pt'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = data.DataLoader(train_dataset, batch_size=conf_common['eval_batch_size'], 
                                   sampler=SequentialSampler(train_dataset), 
                                   num_workers=conf_common['num_workers'])
    eval_loader = data.DataLoader(eval_dataset, batch_size=conf_common['eval_batch_size'], 
                                   sampler=SequentialSampler(eval_dataset), 
                                   num_workers=conf_common['num_workers'])

    # Load final training checkpoint from classifier stage and set network to eval mode
    ckpt = torch.load(os.path.join(classifier_save_path,"model.pth"))
    model = nn.DataParallel(get_network(conf_classifier['network'])(num_classes = 
                            num_classes).to(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Features are extracted either from the output of the backbone network or the 
    # penultimate FC layer (before the classification layer)
    # Store the network features and gt labels of the training set
    X_feats_train = np.zeros((len(train_dataset),model.module.feat_dim if \
                        conf_gen_features['extract_level'] =='backbone' else model.module.fc_dim))
    gt_labels_train = np.zeros(len(train_dataset))
    # Store the network features of the evaluation set
    X_feats_eval = np.zeros((len(eval_dataset),model.module.feat_dim if \
                        conf_gen_features['extract_level'] =='backbone' else model.module.fc_dim))
    # Store the network confidence predictions for the evaluation set
    X_preds_eval = np.zeros((len(eval_dataset),num_classes))
    # Store the gt labels for the evaluation set, used only for calculating cluster metrics
    gt_labels_eval = np.zeros(len(eval_dataset))
    # Store mapping to path of image and corresponding cluster label
    cluster_mapping = [None for i in range(len(eval_dataset))]

    img_count = 0 
    with torch.no_grad():
        # Obtain features for the training set
        for i,(inputs, labels, _) in enumerate(train_loader):
            print_str = "Running train batch "+str(i+1)+" / "+str(len(train_loader))
            sys.stdout.write('\r'+print_str)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.module.features(inputs) # Output of backbone network
            outputs = model.module.avgpool(outputs)
            outputs = outputs.reshape(outputs.size(0), -1)
            # Extract backbone features based on config option
            if conf_gen_features['extract_level'] == 'backbone':
                X_feats_train[img_count:img_count+outputs.size(0),:] = outputs.detach().cpu().numpy()
            outputs = model.module.fc(outputs) # Output of FC layers upto penultimate layer
            if conf_gen_features['extract_level'] == 'fc':
                X_feats_train[img_count:img_count+outputs.size(0),:] = outputs.detach().cpu().numpy()
            gt_labels_train[img_count:img_count+outputs.size(0)] = labels.detach().cpu().numpy()
            img_count = img_count + outputs.size(0)

        assert img_count == len(train_dataset)
        sys.stdout.write(f'\rGenerated {img_count} train features'+' '*20+'\n')

        # Obtain features for evaluation set
        img_count = 0
        for i,(inputs, labels, cur_paths) in enumerate(eval_loader):
            print_str = "Running eval batch "+str(i+1)+" / "+str(len(eval_loader))
            sys.stdout.write('\r'+print_str)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Same as training set for extracting features and gt labels
            outputs = model.module.features(inputs)
            outputs = model.module.avgpool(outputs)
            outputs = outputs.reshape(outputs.size(0), -1)
            if conf_gen_features['extract_level'] == 'backbone':
                X_feats_eval[img_count:img_count+outputs.size(0),:] = outputs.detach().cpu().numpy()
            outputs = model.module.fc(outputs)
            if conf_gen_features['extract_level'] == 'fc':
                X_feats_eval[img_count:img_count+outputs.size(0),:] = outputs.detach().cpu().numpy()
            outputs = model.module.classifier(outputs)
            # Obtain class confidence predictions from network
            X_preds_eval[img_count:img_count+outputs.size(0),:] = outputs.detach().cpu().numpy()
            gt_labels_eval[img_count:img_count+outputs.size(0)] = labels.detach().cpu().numpy()
            # Cluster labels for undiscovered images is set to -1 to imply nondiscovered set
            # Cluster labels for discovered images is obtained from refine stage in previous iter 
            cur_labeling = [(p,discovered_clusters[p] \
                            if p in discovered_clusters else -1) for p in cur_paths]
            cluster_mapping[img_count:img_count+outputs.size(0)] = cur_labeling
            img_count = img_count + outputs.size(0)

        assert img_count == len(eval_dataset)
        sys.stdout.write(f'\rGenerated {img_count} eval features'+' '*20+'\n')
        # Save the features, labels and paths to both train and eval set for subsequent iters/stages
        torch.save({
                    'X_feats_train': X_feats_train, 
                    'Y_train': gt_labels_train,
                    'X_feats_eval': X_feats_eval, 
                    'X_preds_eval': X_preds_eval, 
                    'Y_eval': gt_labels_eval, 
                    'cluster_mapping': cluster_mapping
                    },
                    os.path.join(features_save_path,'features.pt')
                 )

    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(features_save_path,"config.yaml"))