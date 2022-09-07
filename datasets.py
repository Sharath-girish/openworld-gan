from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import torch
import os.path
import sys
import re
import random
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    assert get_image_backend() != 'accimage'
    return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) 

def make_combined_dataset(root_dir, train_classes, discovered_clusters, classes_to_idx, extensions=None, is_valid_file=None):

    images = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions) and os.path.exists(x)
        def get_extension(x):
            return [ext for ext in extensions if x.endswith(extensions)][0]

    class_lengths = {}
    class_names = []
    for target in sorted(classes_to_idx.keys()):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    if target in train_classes:
                        if target not in class_lengths:
                            class_lengths[target] = 0
                            class_names.append(target)
                        class_lengths[target] = class_lengths[target] + 1
                        item = (path, len(class_names)-1)
                        images.append(item)
    assert len(class_names) == len(train_classes)
    num_seen_classes = len(class_names)

    new_class_names = []
    for path, cl_label in discovered_clusters:
        if cl_label < num_seen_classes:
            target = train_classes[cl_label]
            assert target in class_lengths
        else:
            target = str(cl_label-num_seen_classes)
            if target not in class_lengths:
                class_lengths[target] = 0
                new_class_names.append(target)
        class_lengths[target] += 1
        item = (path,cl_label)
        images.append(item)

    return images, class_lengths, class_names+sorted(new_class_names)


class DatasetFolder(VisionDataset):

    def __init__(self, train_root, loader, train_classes, discovered_clusters=[], extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, cache_path=None):
        super(DatasetFolder, self).__init__(train_root, transform=transform,
                                            target_transform=target_transform)
        classes_to_idx = self._find_classes(self.root)
        if not (cache_path and os.path.exists(cache_path)):
            samples, class_lengths, class_names = make_combined_dataset(train_root, train_classes, discovered_clusters, 
                                                    classes_to_idx, extensions, is_valid_file)
            if cache_path and os.path.exists(os.path.dirname(cache_path)):
                torch.save({'samples':samples, 'lengths':class_lengths, 'names':class_names}, cache_path)
        else:
            print('Loading cached dataset at', cache_path)
            dataset_cache = torch.load(cache_path)
            samples = dataset_cache['samples']
            class_lengths =  dataset_cache['lengths']
            class_names = dataset_cache['names']

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.class_names = class_names
        self.samples = samples
        self.class_lengths = class_lengths
        self.class_names = class_names
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes_to_idx

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


    def __len__(self):
        return len(self.samples)



class ImageFolder(DatasetFolder):

    def __init__(self, train_root, train_classes, discovered_clusters=[], transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, cache_path=None):
        super(ImageFolder, self).__init__(train_root, loader, train_classes, discovered_clusters, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          cache_path=cache_path)
        self.imgs = self.samples
