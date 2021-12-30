import json
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import os
import torch
import torch.nn as nn
torch.manual_seed(0)
# from utils.dataset_parser.dataset_processing import prepare_dataset, count_dataset


class Place365(torch.utils.data.Dataset):
    # 73000= 200 * 365
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.base_path = os.path.join(root, "places365/")
        self.filename_path = os.path.join(
            self.base_path, "image_index_label_list.json")
        self.filename_list = json.loads(open(self.filename_path).read())
        # [["./data/0_30_1.jpg", 30, 1], ["./data/1_30_1.jpg", 30, 1], ["./data/2_30_1.jpg", 30, 1], [image_path, category, indoor/outdoor]

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):

        image_path = os.path.join(
            self.base_path, self.filename_list[index][0][2:])
        image = Image.open(image_path).convert('RGB')

        target = {"category": self.filename_list[index][1],
                  "io": self.filename_list[index][2]}

        if self.transform:
            image = self.transform(image)

        return image, target


class Place100(torch.utils.data.Dataset):
    # 73000= 200 * 365
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.base_path = os.path.join(root, "place100/")
        self.filename_path = os.path.join(
            self.base_path, "image_index_label_list.json")
        self.filename_list = json.loads(open(self.filename_path).read())
        # [["place100/data/0_30_1.jpg", 30, 1], ["place100/data/1_30_1.jpg", 30, 1], ["place100/data/2_30_1.jpg", 30, 1], [image_path, category, indoor/outdoor]
        self.category_label_index_dict = {}
        selected_label = set([row[1] for row in self.filename_list])
        index = 0
        for original_label in selected_label:
            self.category_label_index_dict[original_label] = index
            index += 1

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):

        # image_path = os.path.join(self.base_path, self.filename_list[index][0][2:])
        image_path = os.path.join(self.root, self.filename_list[index][0])
        image = Image.open(image_path).convert('RGB')

        target = {"category": self.category_label_index_dict[self.filename_list[index][1]],
                  "io": self.filename_list[index][2]}

        if self.transform:
            image = self.transform(image)

        return image, target


class Place50(torch.utils.data.Dataset):
    # 73000= 200 * 365
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.base_path = os.path.join(root, "place50/")
        self.filename_path = os.path.join(
            self.base_path, "image_index_label_list.json")
        self.filename_list = json.loads(open(self.filename_path).read())
        self.category_label_index_dict = {}
        selected_label = set([row[1] for row in self.filename_list])
        index = 0
        for original_label in selected_label:
            self.category_label_index_dict[original_label] = index
            index += 1

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):

        # image_path = os.path.join(self.base_path, self.filename_list[index][0][2:])
        image_path = os.path.join(self.root, self.filename_list[index][0])
        image = Image.open(image_path).convert('RGB')

        target = {"category": self.category_label_index_dict[self.filename_list[index][1]],
                  "io": self.filename_list[index][2]}

        if self.transform:
            image = self.transform(image)

        return image, target


class Place20(torch.utils.data.Dataset):
    # 73000= 200 * 365
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.base_path = os.path.join(root, "place20/")
        self.filename_path = os.path.join(
            self.base_path, "image_index_label_list.json")
        self.filename_list = json.loads(open(self.filename_path).read())
        self.category_label_index_dict = {}
        selected_label = set([row[1] for row in self.filename_list])
        index = 0
        for original_label in selected_label:
            self.category_label_index_dict[original_label] = index
            index += 1

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):

        # image_path = os.path.join(self.base_path, self.filename_list[index][0][2:])
        image_path = os.path.join(self.root, self.filename_list[index][0])
        image = Image.open(image_path).convert('RGB')

        target = {"category": self.category_label_index_dict[self.filename_list[index][1]],
                  "io": self.filename_list[index][2]}

        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    root = "/home/xinlei.he/simclr/data/"
    dataset = Place20(root=root, transform=transform)
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(len(dataset))
    print(dataset.category_label_index_dict)

    target_train, target_test, shadow_train, shadow_test = prepare_dataset(
        dataset)
    target_train_loader = torch.utils.data.DataLoader(
        target_train, batch_size=32, shuffle=True, num_workers=2)
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=32, shuffle=True, num_workers=2)
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=32, shuffle=True, num_workers=2)
    shadow_test_laoder = torch.utils.data.DataLoader(
        shadow_test, batch_size=32, shuffle=True, num_workers=2)
    all_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=2)
    count_dataset(target_train_loader, target_test_loader,
                  shadow_train_loader, shadow_test_laoder, num_classes=2, attr="io")
