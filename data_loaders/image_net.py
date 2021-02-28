import os
import numpy as np
import matplotlib.image as mpimg
from typing import List
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """
    A DataLoader for ImageNet Dataset designed specifically for KU HPC Cluster.
    """
    labels_dir = '/datasets/ImageNet/ILSVRC2015/devkit/data'
    train_labels_file = 'map_clsloc.txt'
    valid_labels_file = 'ILSVRC2015_clsloc_validation_ground_truth.txt'
    train_labels_path = os.path.join(labels_dir, train_labels_file)
    valid_labels_path = os.path.join(labels_dir, valid_labels_file)
    train_data_dir = '/datasets/ImageNet/ILSVRC2015/Data/CLS-LOC/train'
    valid_data_dir = '/datasets/ImageNet/ILSVRC2015/Data/CLS-LOC/val'
    
    def __init__(self, train: bool, num_classes: int = 1000, transform=None, class_list: List[int] = None):

        self.transform = transform
        
        self.class_id_to_index = {}
        self.class_index_to_name = {}
        self.class_name_to_index = {}
        self.train = train
        self.class_list = class_list
        self.classes = []

        if class_list is not None:
            for label in class_list:
                self.classes.append(self.class_index_to_name[label])
        else:
            for label in range(num_classes):
                self.classes.append(self.class_index_to_name[label])

        # store labels
        self.labels = []
        # store paths to images
        self.data = []
        
        # get classed mapping
        train_f = open(self.train_labels_path, "r") 
        for line in train_f:
            ID, index, name = line[:-1].split(' ')
            index = int(index) - 1
            self.class_id_to_index[ID] = index
            self.class_index_to_name[index] = name
            self.class_name_to_index[name] = index
        
        if train:
            train_data_folders = os.listdir(self.train_data_dir)
            for ID in train_data_folders:
                label = self.class_id_to_index[ID]

                if class_list is not None and label not in class_list:
                    continue
                elif label > num_classes:
                    continue
                
                label_path = os.path.join(self.train_data_dir, ID)
                for data_file in os.listdir(label_path):
                    self.data.append(os.path.join(label_path, data_file))
                    self.labels.append(label)
        else:
            valid_f = open(self.valid_labels_path, "r")
            mask = []
            for line in valid_f:
                label = int(line[:-1])
                if class_list is not None and label not in class_list:
                    mask.append(False)
                    continue
                elif label > num_classes:
                    mask.append(False)
                    continue

                self.labels.append(label)
                mask.append(True)
            i = 0
            for data_file in sorted(os.listdir(self.valid_data_dir)):
                if not mask[i]:
                    i += 1
                    continue
                self.data.append(os.path.join(self.valid_data_dir, data_file))
                i += 1
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        
        label = self.labels[index]
        data_path = self.data[index]
        data = mpimg.imread(data_path)
        
        # some images might have 1 channel
        if len(data.shape) < 3:
            H, W = data.shape
            data_temp = data.copy()
            data = np.zeros((H, W, 3))
            for i in range(3):
                data[:, :, i] = data_temp
            
            data = data.astype(np.uint8)
        if self.transform:
            data = self.transform(data)
        
        return data, label

    def get_class_list(self):
        return self.class_list

    def get_class_name(self, label):
        return self.class_index_to_name[label]
