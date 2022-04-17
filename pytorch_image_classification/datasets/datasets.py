from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config

from torch.utils.data import Dataset

import PIL
from pytorch_image_classification import create_transform

def pil_grayscale_loader(path):
    with open(path, 'rb') as f:
        image = PIL.Image.open(f)
        return image.copy()

class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',


    ]:


        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            # if config.train.use_test_as_val == 0:
            #     return 0
            if config.train.use_test_as_val:
                # return 0
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir_val,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                
                # print('**********************************')
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                       transform=val_transform)
        return train_dataset, val_dataset

    elif config.dataset.name == 'SAR10':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        dataset_dir_val = pathlib.Path(config.dataset.dataset_dir_val).expanduser()

        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)


        if config.train.use_test_as_val:
            train_dataset = torchvision.datasets.ImageFolder(dataset_dir, loader=pil_grayscale_loader, transform=train_transform)
            test_dataset = torchvision.datasets.ImageFolder(dataset_dir_val, loader=pil_grayscale_loader, transform=val_transform)
            print('**********************************')
            return train_dataset, test_dataset
        else:
            dataset = torchvision.datasets.ImageFolder(dataset_dir, loader=pil_grayscale_loader)
            # dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=train_transform)#image 读进来还正常

            val_ratio = config.train.val_ratio
            assert val_ratio < 1
            val_num = int(len(dataset) * val_ratio)
            train_num = len(dataset) - val_num
            lengths = [train_num, val_num]
            train_subset, val_subset = torch.utils.data.dataset.random_split(
                dataset, lengths)

            train_dataset = SubsetDataset(train_subset, train_transform)#这里不正常了
            val_dataset = SubsetDataset(val_subset, val_transform)
            return train_dataset, val_dataset
            # return train_subset, val_subset
    else:
        raise ValueError()






if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import pytorch_image_classification

    config = pytorch_image_classification.get_default_config()
    config.merge_from_file('../../configs/sar10/resnet.yaml')
    # print(config)
    # dataset1, dataset2 = create_dataset(config, is_train=True)
    # #
    # img = dataset1.__getitem__(3)[0]
    # print(f'img shape{img.shape}, img type {img.dtype}')
    # img = img.permute(1,2,0).numpy()
    # print(f'img shape{img.shape}, img type {img.dtype}')
    # #
    # plt.imshow(img)
    # plt.show()
    img_path = '/home/lixiaohan/Dataset/SAR/SAR_test/SAR_244010.png'
    img_test = PIL.Image.open(img_path).convert('L')

    train_transform = create_transform(config, is_train=True)
    out1 = train_transform(img_test)
    out1 = out1.numpy().squeeze()
    plt.imshow(out1)
    plt.show()


    # img_test1 = img_test.filter(PIL.ImageFilter.Filter())
    # plt.title('DETAIL ')
    # img_test1 = img_test.filter(PIL.ImageFilter.DETAIL)
    # plt.imshow(img_test1)
    # plt.figure()
    #
    # plt.title('EDGE_ENHANCE ')
    # img_test2 = img_test.filter(PIL.ImageFilter.EDGE_ENHANCE)
    # plt.imshow(img_test2)
    # plt.figure()
    #
    # plt.title('SHARPEN ')
    # img_test3 = img_test.filter(PIL.ImageFilter.SHARPEN)
    # plt.imshow(img_test3)
    # plt.figure()
    # plt.title('ORIGINAL')
    # plt.imshow(img_test)
    # plt.figure()
    #
    # img_test4 = PIL.ImageEnhance.Sharpness(img_test).enhance(2)
    # # print(img_test4.size())
    # plt.imshow(img_test4)
    # out = PIL.ImageFilter.EDGE_ENHANCE(img_test)
    # plt.show()




