import torch
import torch.nn as nn
torch.manual_seed(0)


def count_dataset(targetTrainloader, targetTestloader, shadowTrainloader, shadowTestloader, num_classes, attr=None):
    target_train = [0 for i in range(num_classes)]
    target_test = [0 for i in range(num_classes)]
    shadow_train = [0 for i in range(num_classes)]
    shadow_test = [0 for i in range(num_classes)]

    for _, num in targetTrainloader:
        if attr != None:
            num = num[attr]
        for row in num:
            target_train[int(row)] += 1

    for _, num in targetTestloader:
        if attr != None:
            num = num[attr]
        for row in num:
            target_test[int(row)] += 1

    for _, num in shadowTrainloader:
        if attr != None:
            num = num[attr]
        for row in num:
            shadow_train[int(row)] += 1

    for _, num in shadowTestloader:
        if attr != None:
            num = num[attr]
        for row in num:
            shadow_test[int(row)] += 1

    print(target_train)
    print(target_test)
    print(shadow_train)
    print(shadow_test)


def prepare_dataset(dataset):

    length = len(dataset)
    each_length = length//4

    torch.manual_seed(0)
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    return target_train, target_test, shadow_train, shadow_test


def cut_dataset(dataset, num):

    length = len(dataset)

    torch.manual_seed(0)
    seleted_dataset, _ = torch.utils.data.random_split(
        dataset, [num, length - num])
    return seleted_dataset
