import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import random


class attackTraining():
    def __init__(self, opt, targetTrainloader, targetTestloader, shadowTrainloader, shadowTestloader, target_model, shadow_model, attack_model, device):
        self.device = device
        self.opt = opt

        self.target_model = target_model.to(self.device).eval()
        self.shadow_model = shadow_model.to(self.device).eval()
        if self.opt.select_posteriors == -1:
            self.selected_posterior = 2
        else:
            self.selected_posterior = self.opt.select_posteriors

        self.targetTrainloader = targetTrainloader
        self.targetTestloader = targetTestloader
        self.shadowTrainloader = shadowTrainloader
        self.shadowTestloader = shadowTestloader

        self.attack_model = attack_model.to(self.device)

        self.attack_model.apply(self._weights_init_normal)

        self.optimizer = torch.optim.Adam(
            self.attack_model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.original_performance = [0.0, 0.0, 0.0, 0.0]

    def parse_dataset(self):
        self.getTrainDataset()
        self.getTestDataset()

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

    def get_label(self, label):
        if self.opt.dataset in self.opt.single_label_dataset:
            return label
        elif self.opt.dataset in self.opt.multi_label_dataset:
            return label[self.opt.original_label]
        else:
            raise ValueError("dataset not found")

    def get_data(self, dataloader, model_type="target", member_type=1):
        data = []
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), desc="process"):

                # targets = self.get_target(targets)
                inputs, targets = inputs.to(
                    self.device), self.get_label(targets).to(self.device)
                if model_type == "shadow":
                    _, outputs = self.shadow_model(inputs)
                elif model_type == "target":
                    _, outputs = self.target_model(inputs)

                # if self.opt.mia_defense == "MemGuard":
                #     from utils.mia.memguard import MemGuard
                #     memGuard = MemGuard()
                #     outputs = memGuard(outputs)
                # else:
                # # print("before softmax:", outputs)
                outputs = F.softmax(outputs, dim=1)
                # print("after softmax:", outputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if torch.cuda.is_available():
                    outputs = outputs.cpu()

                outputs = outputs.numpy().tolist()

                s = predicted.eq(targets).cpu().numpy().tolist()
                for i in range(len(outputs)):
                    outputs[i].sort(reverse=True)
                    data.append(
                        [outputs[i][:self.selected_posterior] + [float(s[i])], member_type])
        print("posterior mean: ", np.mean([row[0] for row in data], axis=0))
        print("acc: %.3f" % (correct / total))
        if model_type == "target" and member_type == 1:  # target train
            self.original_performance[0] = 1.0 * correct / total
        elif model_type == "target" and member_type == 0:  # target test
            self.original_performance[1] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 1:  # target train
            self.original_performance[2] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 0:  # target test
            self.original_performance[3] = 1.0 * correct / total

        return data

    def getTrainDataset(self):
        mem = []
        non_mem = []
        attack_set = []
        attack_label = []
        print("shadow mem")
        mem = self.get_data(self.shadowTrainloader,
                            model_type="shadow", member_type=1)
        print("shadow nomem")
        non_mem = self.get_data(self.shadowTestloader,
                                model_type="shadow", member_type=0)
        train_set = mem + non_mem
        for train_data, train_label in train_set:
            attack_set.append(train_data)
            attack_label.append(train_label)
        # print(len(attack_label), attack_label.count(0), attack_label.count(1))
        train = torch.utils.data.TensorDataset(torch.from_numpy(np.array(
            attack_set, dtype='f')), torch.from_numpy(np.array(attack_label)).type(torch.long))
        self.attack_train_loader = torch.utils.data.DataLoader(
            train, batch_size=256, shuffle=True)

    def getTestDataset(self):
        mem = []
        non_mem = []
        attack_set = []
        attack_label = []
        print("target mem")
        mem = self.get_data(self.targetTrainloader,
                            model_type="target", member_type=1)
        print("target nomem")
        non_mem = self.get_data(self.targetTestloader,
                                model_type="target", member_type=0)
        train_set = mem + non_mem
        for train_data, train_label in train_set:
            attack_set.append(train_data)
            attack_label.append(train_label)
        # print(len(attack_label), attack_label.count(0), attack_label.count(1))
        test = torch.utils.data.TensorDataset(torch.from_numpy(np.array(
            attack_set, dtype='f')), torch.from_numpy(np.array(attack_label)).type(torch.long))
        self.attack_test_loader = torch.utils.data.DataLoader(
            test, batch_size=256, shuffle=True)

    def train(self, train_epoch):

        for e in range(train_epoch):
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

        return 1.*correct/total
