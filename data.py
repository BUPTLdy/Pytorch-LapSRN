#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:42:16 2017

@author: ldy
"""

from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip

from dataset import DatasetFromFolder

crop_size =128
def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir




    

def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//8),
        ToTensor(),
    ])

def HR_2_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])

def HR_4_transform(crop_size):
    return Compose([
        Scale(crop_size//2),
        ToTensor(),
    ])

def HR_8_transform(crop_size):
    return Compose([
        RandomCrop((crop_size, crop_size)),
        RandomHorizontalFlip(),
    ])


def get_training_set():
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             LR_transform=LR_transform(crop_size),
                             HR_2_transform=HR_2_transform(crop_size),
                             HR_4_transform=HR_4_transform(crop_size),
                             HR_8_transform=HR_8_transform(crop_size))


def get_test_set():
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             LR_transform=LR_transform(crop_size),
                             HR_2_transform=HR_2_transform(crop_size),
                             HR_4_transform=HR_4_transform(crop_size),
                             HR_8_transform=HR_8_transform(crop_size))

