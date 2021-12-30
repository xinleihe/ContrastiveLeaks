from utils.dataset_parser.dataset_processing import prepare_dataset, count_dataset
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import os
import torch
import torch.nn as nn
torch.manual_seed(0)


class UTKFace(torch.utils.data.Dataset):
    # 23707
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.lines = [row for row in os.listdir(
            root+'/UTKFace/') if "jpg" in row]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        attrs = self.lines[index].split('_')

        if len(attrs) != 4:
            print("Wrong!", self.lines[index])
        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(
            self.root+'/UTKFace/', self.lines[index]).rstrip()

        image = Image.open(image_path).convert('RGB')

        target = {"Age": age,
                  "Gender": gender,
                  "Race": race}

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
    dataset = UTKFace(root=root, transform=transform)
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
                  shadow_train_loader, shadow_test_laoder, num_classes=2, attr="Gender")
    count_dataset(target_train_loader, target_test_loader,
                  shadow_train_loader, shadow_test_laoder, num_classes=5, attr="Race")
    # Supervised
    # [3064, 2862]
    # [3068, 2858]
    # [3128, 2798]
    # [3131, 2795]
    # Unsupervised
    # [3044, 2844]
    # [3047, 2841]
    # [3111, 2777]
    # [3106, 2782]
    # for x,y in all_data_loader:
    #     print(x, y)
