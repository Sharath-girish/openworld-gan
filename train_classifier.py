from builtins import breakpoint
import os
import sys
import time
import utils
import torch
import config
import torch.nn
import numpy as np
import torch.nn as nn
from config import Config
import torch.utils.data as data
from datasets import ImageFolder
from networks import get_network
from transforms import get_transform
 
def train(step, conf, save_conf):
    # Obtain config for hyperparameters of current stage
    conf_common = config.get_conf_common(conf)
    root_path = conf_common['save_path']
    conf_step = config.get_conf_step(conf,step)
    conf_classifier = config.get_conf_classifier(conf_step)
    # Save path for current classifier stage
    classifier_save_path = utils.get_classifier_path(root_path,step)

    # Obtain list of train classes
    classes_list = Config(config_filepath='class_list.yaml')
    train_classes = sorted(classes_list[conf_classifier['train_classes']]['train_classes'])

    # No previous discovered clusters if it is the first iteration
    if step == 1:
        discovered_clusters = []
    else:
        # Obtain the discovered cluster labels from refine stage of previous iteration
        # discovered_clusters contains the full list of evaluation images which are 2-tuples
        # with the format (<path to image>, cluster label)
        refine_save_path = utils.get_refine_path(root_path,step-1)
        num_seen_classes = len(train_classes) # Number of classes with training gt labels
        features = torch.load(os.path.join(refine_save_path,"features.pt"))
        discovered_clusters = []
        for item in features['cluster_mapping']:
            # Do not include samples in the non-discovered set with cluster label set to -1
            # Do not include samples with labels belonging to seen classes for training if
            # conf_classifier['add_seen'] is false
            # This is done as training set already contains seen class images with pure gt labels
            if item[1] >= num_seen_classes if not conf_classifier['add_seen'] else 0:
                discovered_clusters.append(item)

    # Create relevant data transforms, dataset and dataloader
    transform = get_transform(conf_classifier['transform'], resize_dims=conf_classifier['resize'])
    train_data_path = conf_common['train_data_path']
    train_dataset = ImageFolder(train_data_path, train_classes, discovered_clusters, transform=transform, 
                                cache_path=os.path.join(classifier_save_path,'train_dataset_cache.pt'))
    # Obtain number of images in each class (/cluster) for weighing cross entropy loss
    class_lengths = torch.Tensor([train_dataset.class_lengths[c] for c in \
                                    train_dataset.class_names])
    train_loader = data.DataLoader(train_dataset, batch_size=conf_classifier['batch_size'], 
                                    shuffle=True, num_workers=conf_common['num_workers'])

    # Create model for n-way classification based on number of clusters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(get_network(conf_classifier['network'])(num_classes = len(class_lengths))\
            .to(device))

    # Set class weights based on number of images in each class
    class_weights = class_lengths.max()/class_lengths
    if 'weighted' in conf_classifier and conf_classifier['weighted']:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf_classifier['learning_rate'])

    start_epoch = 0

    # Load weights from model trained in previous iteration of the pipeline
    if 'pretrained' in conf_classifier and conf_classifier['pretrained'] and step>1:
        ckpt = torch.load(os.path.join(utils.get_classifier_path(root_path,step-1),"model.pth"))
        load_state_dict = ckpt['model_state_dict']
        model_state_dict = model.state_dict()
        for key in model_state_dict:
            if 'classifier' in key:
                load_state_dict[key] = model_state_dict[key]
        model.load_state_dict(load_state_dict)
        print(f'Loaded model weights from step {step-1}')

    # Load weights from existing checkpoint for resuming, if exists
    if os.path.exists(os.path.join(classifier_save_path,"checkpoint.pth")):
        ckpt = torch.load(os.path.join(classifier_save_path,"checkpoint.pth"))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt['epoch']
        print('Resuming model training from epoch {}'.format(start_epoch+1))

    total_step = len(train_loader)

    # Main training loop
    for epoch in range(start_epoch, conf_classifier['num_epochs']):
        st = time.time()
        model.train()

        epoch_loss = correct = total = 0.0

        for i, (images, labels, _) in enumerate(train_loader):
            sys.stdout.write('\r{}/{}'.format(i+1,total_step))
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        sys.stdout.write('\r'+'Epoch [{}/{}], Epoch Loss: {:.4f}\n'.format(epoch+1, conf_classifier['num_epochs'], epoch_loss/(i+1)))
        print('Train Accuracy of the model: {:.4f} %'.format(100 * correct / total))
        print('Time taken for training epoch {}: {:.4f} s'.format(epoch+1,time.time()-st))

        # Checkpoint every epoch for resuming
        torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss/(i+1),
                    }, os.path.join(classifier_save_path,"checkpoint.pth"))

    # Checkpoint final model at the end of training
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/(i+1),
                }, os.path.join(classifier_save_path,"model.pth"))
    
    # Save config to signal end of current stage
    config._save_to_file(save_conf,os.path.join(classifier_save_path,"config.yaml"))