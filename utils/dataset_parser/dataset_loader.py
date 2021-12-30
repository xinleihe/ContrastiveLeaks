from utils.dataset_parser.dataset_processing import prepare_dataset, cut_dataset, count_dataset
from utils.dataset_parser.place365_parser import Place365, Place100, Place50, Place20
from utils.dataset_parser.utkface_parser import UTKFace
from utils.dataset_parser.celeba_parser import CelebA
from utils.dataset_parser.gaussian_blur import GaussianBlur
from torchvision import datasets
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
torch.manual_seed(0)


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class AdvSimCLRDataTransform(object):
    def __init__(self, transform, original_transform):
        self.transform = transform
        self.original_transform = original_transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        x = self.original_transform(sample)
        return xi, xj, x


class GetDataLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.data_path = self.opt.data_path
        # constant value
        self.s = 1
        self.input_shape = (96, 96, 3)

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(self.input_shape[0])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(kernel_size=int(
                                                  0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, train_transform, test_transform):

        if self.opt.dataset == "CIFAR10":
            train_dataset = datasets.CIFAR10(self.data_path,
                                             train=True,
                                             transform=train_transform,
                                             download=False,)
            test_dataset = datasets.CIFAR10(self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=False,)
            dataset = train_dataset + test_dataset

        elif self.opt.dataset == "CIFAR100":
            train_dataset = datasets.CIFAR100(root=self.data_path,
                                              train=True,
                                              transform=train_transform,
                                              download=False)
            test_dataset = datasets.CIFAR100(root=self.data_path,
                                             train=False,
                                             transform=test_transform,
                                             download=False)
            dataset = train_dataset + test_dataset

        elif self.opt.dataset == "STL10":
            unlabel_dataset = datasets.STL10(self.data_path,
                                             split='unlabeled',
                                             transform=train_transform,
                                             download=False,)
            train_dataset = datasets.STL10(self.data_path,
                                           split='train',
                                           transform=train_transform,
                                           download=False,)
            test_dataset = datasets.STL10(self.data_path,
                                          split='test',
                                          transform=test_transform,
                                          download=False,)
            dataset = train_dataset + test_dataset

        elif self.opt.dataset == "CelebA":
            dataset = CelebA(self.data_path, transform=train_transform)
            # randomly select 30,000 Images from CelebA dataset
            dataset = cut_dataset(dataset, num=30000)

        elif self.opt.dataset == "UTKFace":
            dataset = UTKFace(self.data_path, transform=test_transform)

        elif self.opt.dataset == "Place365":
            dataset = Place365(self.data_path, transform=test_transform)

        elif self.opt.dataset == "Place100":
            dataset = Place100(self.data_path, transform=test_transform)
        elif self.opt.dataset == "Place50":
            dataset = Place50(self.data_path, transform=test_transform)
        elif self.opt.dataset == "Place20":
            dataset = Place20(self.data_path, transform=test_transform)
        return dataset

    def get_data_supervised(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[0])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[0])),
            transforms.ToTensor()])
        dataset = self.get_dataset(train_transform, test_transform)
        target_train, target_test, shadow_train, shadow_test = prepare_dataset(
            dataset)
        print("Preparing dataloader!")
        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)

        # all_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)

        # print("Preparing dataloader statistic!")
        # if self.opt.dataset == "UTKFace":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="Gender")
        # elif self.opt.dataset == "Place100":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="io")
        # elif self.opt.dataset == "Place50":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="io")
        # elif self.opt.dataset == "Place20":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="io")
        # exit()

        return target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader

    def get_STL_pretrain(self):
        # data_augment = self._get_simclr_pipeline_transform()
        train_transform = SimCLRDataTransform(
            self._get_simclr_pipeline_transform())
        unlabel_dataset = datasets.STL10(self.data_path,
                                         split='unlabeled',
                                         transform=train_transform,
                                         download=False,)
        target_train_loader = torch.utils.data.DataLoader(
            unlabel_dataset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        return target_train_loader, None, None, None

    def get_data_unsupervised(self):
        # data_augment = self._get_simclr_pipeline_transform()
        train_transform = SimCLRDataTransform(
            self._get_simclr_pipeline_transform())
        test_transform = SimCLRDataTransform(
            self._get_simclr_pipeline_transform())
        dataset = self.get_dataset(train_transform, test_transform)
        target_train, target_test, shadow_train, shadow_test = prepare_dataset(
            dataset)
        print("Preparing dataloader!")
        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        # all_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)

        return target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader

    def get_data_unsupervised_adv(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_original_transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[0])),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()])
        test_original_transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[0])),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

        train_transform = AdvSimCLRDataTransform(
            transform=data_augment, original_transform=train_original_transform)
        test_transform = AdvSimCLRDataTransform(
            transform=data_augment, original_transform=test_original_transform)
        dataset = self.get_dataset(train_transform, test_transform)
        target_train, target_test, shadow_train, shadow_test = prepare_dataset(
            dataset)
        print("Preparing dataloader!")
        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=self.opt.num_workers)
        all_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)

        # print("Preparing dataloader statistic!")
        # if self.opt.dataset == "UTKFace":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="Gender")
        # elif self.opt.dataset == "CelebA":
        #     count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, self.opt.n_class, attr="Male")

        return target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader
