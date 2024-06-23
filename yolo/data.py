import os
import cv2
import glob
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from zipfile import ZipFile

# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set='train', zip_file=None, data_path='data'):
        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        if zip_file is not None:
            with ZipFile(zip_file, 'r') as zObject:
                zObject.extractall(path="")
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)

    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):

        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))

        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]

        return gt_files_train, gt_files_val

    def __getitem__(self, idx):

        # gt file
        filename_gt = self.gt_paths[idx]

        # Computing image_blob
        img_filename = filename_gt[:-len('-box.txt')] + '.jpg'
        img = cv2.imread(filename=img_filename)
        dim = (self.yolo_image_size, self.yolo_image_size)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        image_blob = (img - self.pixel_mean) / 255.0
        image_blob = torch.tensor(image_blob.transpose((2, 0, 1)))

        # Computing gt_box_blob
        dim = (5, self.yolo_grid_num, self.yolo_grid_num)
        gt_box_blob = torch.zeros(dim)

        # Get the ground truth values from the file
        string_ = open(filename_gt, 'r').read()
        x1, y1, x2, y2 = np.fromstring(string_, dtype=float, sep=' ')
        x1_scaled = x1 * self.scale_width
        x2_scaled = x2 * self.scale_width
        y1_scaled = y1 * self.scale_height
        y2_scaled = y2 * self.scale_height
        w = x2_scaled - x1_scaled
        h = y2_scaled - y1_scaled
        cx = (x1_scaled+x2_scaled) / 2
        cy = (y1_scaled+y2_scaled) / 2
        c = np.array((cy, cx))
        cell = (c//self.yolo_grid_size).astype(int)
        offset = torch.tensor(c - cell*self.yolo_grid_size)
        normalized_offset = offset / self.yolo_grid_size

        gt_box_blob[0, cell[0], cell[1]] = normalized_offset[1]
        gt_box_blob[1, cell[0], cell[1]] = normalized_offset[0]
        gt_box_blob[2, cell[0], cell[1]] = w / self.yolo_image_size
        gt_box_blob[3, cell[0], cell[1]] = h / self.yolo_image_size
        gt_box_blob[4, cell[0], cell[1]] = 1

        dim = (self.yolo_grid_num, self.yolo_grid_num)
        gt_mask_blob = torch.zeros(dim)
        gt_mask_blob[cell[0], cell[1]] = 1

        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample

    # len of the dataset
    def __len__(self):
        return self.size

