from tqdm import tqdm
import pandas as pd
from functools import partial
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import os
import torch
import torch.nn as nn
torch.manual_seed(0)


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.root = root
        self.transform = transform
        self.base_folder = "celeba"

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pd.read_csv(fn("list_eval_partition.txt"),
                             delim_whitespace=True, header=None, index_col=0)
        # identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        # landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        self.attr = pd.read_csv(
            fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        # change label -1 to label 0
        self.attr = self.attr.replace(-1, 0)

        # mask = slice(None)
        # filename: ['000001.jpg' '000002.jpg' '000003.jpg' ... '202597.jpg' '202598.jpg' '202599.jpg']
        self.filename = splits.index.values

        self.attr_names = list(self.attr.columns)
        # self.__preprocessing__()
        # print(self.attr_names, len(self.attr_names))
        # print(self.attr)
        # print(self.attr.loc[self.filename[0]])

    # def __preprocessing__(self):

    #     self.image_list = []
    #     for index in tqdm(range(len(self.filename)), desc="load image"):
    #         self.image_list.append(Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])))

    def __getitem__(self, index):
        X = Image.open(os.path.join(self.root, self.base_folder,
                       "img_align_celeba", self.filename[index]))
        # X = jpeg.JPEG(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])).decode()
        # X = self.image_list[index]
        target = {k: self.attr.loc[self.filename[index]][k]
                  for k in self.attr_names}
        target["Mouth_Smile"] = target["Mouth_Slightly_Open"] * \
            2 + target["Smiling"]
        target["Young_Smile"] = target["Young"] * 2 + target["Smiling"]
        target["Attractive_Smile"] = target["Attractive"] * \
            2 + target["Smiling"]

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self):
        return len(self.attr)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    root = "/home/xinlei.he/simclr/data/"
    dataset = CelebA(root=root, transform=transform)
    print(len(dataset))
    count_label = [0, 0, 0, 0]
    sample_dataset, rest_dataset = torch.utils.data.random_split(
        dataset, [5000, len(dataset)-(5000)])
    for target, label in tqdm(sample_dataset):
        index = label["Young_Smile"]
        count_label[index] += 1
    print("label distribution", count_label)
