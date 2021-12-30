import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import random


class attackTraining():
    def __init__(self, opt, aux_train_loader, aux_test_loader, target_model, attack_model, device):
        self.device = device

        self.opt = opt
        self.target_model = target_model.to(self.device).eval()

        self.aux_train_loader = aux_train_loader
        self.aux_test_loader = aux_test_loader

        self.attack_model = attack_model.to(self.device)

        self.attack_model.apply(self._weights_init_normal)

        self.optimizer = torch.optim.SGD(
            self.attack_model.parameters(), lr=self.opt.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.original_performance = [0.0, 0.0]
        self.parse_dataset()

    def _weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

    def get_label(self, label, label_type="aux_label"):
        if self.opt.dataset in self.opt.single_label_dataset:
            return label
        elif self.opt.dataset in self.opt.multi_label_dataset:
            if label_type == "aux_label":
                # Node that for the overlearning task, we use the aux_label as the ground truth
                return label[self.opt.aux_label]
            elif label_type == "original_label":
                return label[self.opt.original_label]
            else:
                raise ValueError("label_type is not correct!")
        else:
            raise ValueError("dataset not found!")

    def parse_dataset(self):
        self.get_train_dataset()
        self.get_test_dataset()

    def get_data(self, dataloader, data_type="train"):
        data = []
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, original_label, aux_label = inputs.to(self.device), \
                    self.get_label(targets, label_type="original_label").to(self.device), \
                    self.get_label(targets, label_type="aux_label").to(
                        self.device)

                embeddings, outputs = self.target_model(inputs)

                # get target task prediction results
                outputs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += original_label.size(0)
                correct += predicted.eq(original_label).sum().item()
                if data_type == "train":
                    self.original_performance[0] = correct / total
                elif data_type == "test":
                    self.original_performance[1] = correct / total

                # get embedding and aux label
                if torch.cuda.is_available():
                    embeddings = embeddings.cpu()
                    aux_label = aux_label.cpu()
                embeddings = embeddings.numpy().tolist()
                aux_label = aux_label.numpy().tolist()
                for i in range(len(embeddings)):
                    data.append([embeddings[i], aux_label[i]])

        return data

    def get_train_dataset(self):
        attack_set = []
        attack_label = []
        train_set = self.get_data(self.aux_train_loader, data_type="train")
        np.random.seed(1)
        np.random.shuffle(train_set)
        seleted_num = int(len(train_set) * self.opt.ratio)
        train_set = train_set[:seleted_num]
        for train_data, train_label in train_set:
            attack_set.append(train_data)
            attack_label.append(train_label)
        train = torch.utils.data.TensorDataset(torch.from_numpy(np.array(
            attack_set, dtype='f')), torch.from_numpy(np.array(attack_label)).type(torch.long))
        self.attack_train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.opt.batch_size, shuffle=True)

    def get_test_dataset(self):
        attack_set = []
        attack_label = []
        train_set = self.get_data(self.aux_test_loader, data_type="test")
        for train_data, train_label in train_set:
            attack_set.append(train_data)
            attack_label.append(train_label)
        test = torch.utils.data.TensorDataset(torch.from_numpy(np.array(
            attack_set, dtype='f')), torch.from_numpy(np.array(attack_label)).type(torch.long))
        self.attack_test_loader = torch.utils.data.DataLoader(
            test, batch_size=self.opt.batch_size, shuffle=True)

    def train(self, train_epoch):

        for e in range(train_epoch):
            if e % 10 == 0:
                test_acc = self.test()
            train_loss = 0
            correct = 0
            total = 0
            self.attack_model.train()
            for inputs, targets in self.attack_train_loader:
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.attack_model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print('Epoch: %d, Train Acc: %.3f%% (%d/%d)' %
                  (e, 100.*correct/total, correct, total))
        test_acc = self.test()
        train_acc = 1.*correct/total

        return train_acc, test_acc

    def test(self):
        self.attack_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.attack_test_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.attack_model(inputs)
                outputs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Test Acc: %.3f%% (%d/%d)' %
                  (100.*correct/total, correct, total))
        self.attack_model.train()
        return 1.*correct/total
