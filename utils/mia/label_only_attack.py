from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier


class AttackLabelOnly():
    def __init__(self, opt, targetTrainloader, targetTestloader,
                 shadowTrainloader, shadowTestloader, target_model,
                 shadow_model, attack_model, device):
        self.opt = opt
        self.epochs = self.opt.epochs
        self.device = device
        self.activation = {}
        self.target_model = target_model.to(self.device).eval()
        self.shadow_model = shadow_model.to(self.device).eval()

        self.targetTrainloader = targetTrainloader
        self.targetTestloader = targetTestloader
        self.shadowTrainloader = shadowTrainloader
        self.shadowTestloader = shadowTestloader

        self.shadowTrain, self.shadowTest, self.TargetTrain, self.TargetTest = None, None, None, None
        self.distance_threshold_tau = None
        self.num_samples = 100

        self.criterion = nn.CrossEntropyLoss()
        self.original_performance = [0.0, 0.0, 0.0, 0.0, 0.0]

    def get_label(self, label):
        if self.opt.dataset in self.opt.single_label_dataset:
            return label
        elif self.opt.dataset in self.opt.multi_label_dataset:
            return label[self.opt.original_label]
        else:
            raise ValueError("dataset not found")

    def get_data(self, dataloader, model_type="target", member_type=1, num_batches=10):
        data = None
        labels = None

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader),
                                                     desc="process"):
                targets = self.get_label(targets)
                data = inputs.numpy() if batch_idx == 0 else np.concatenate(
                    (data, inputs.numpy()), axis=0)
                labels = targets.numpy() if batch_idx == 0 else np.concatenate(
                    (labels, targets.numpy()), axis=0)

                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                if model_type == "shadow":
                    outputs = self.shadow_model(inputs)
                elif model_type == "target":
                    outputs = self.target_model(inputs)
                else:
                    raise ValueError(
                        "model_type should be either target or shadow")

                outputs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print("overall acc: %.3f" % (correct / total))
        if model_type == "target" and member_type == 1:  # target train
            self.original_performance[0] = 1.0 * correct / total
        elif model_type == "target" and member_type == 0:  # target test
            self.original_performance[1] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 1:  # target train
            self.original_performance[2] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 0:  # target test
            self.original_performance[3] = 1.0 * correct / total

        return data, labels

    def parse_dataset(self):
        print("parse_dataset")
        self.getTrainDataset()
        self.getTestDataset()

    def getTrainDataset(self):
        print("shadow mem")
        mem, mem_target_labels = self.get_data(self.shadowTrainloader,
                                               model_type="shadow",
                                               member_type=1)
        print("shadow nomem")
        non_mem, non_mem_target_labels = self.get_data(self.shadowTestloader,
                                                       model_type="shadow",
                                                       member_type=0)

        self.shadowTrain = (mem, mem_target_labels)
        self.shadowTest = (non_mem, non_mem_target_labels)

    def getTestDataset(self):

        print("target mem")

        mem, mem_target_labels = self.get_data(self.targetTrainloader,
                                               model_type="target",
                                               member_type=1)
        print("target nomem")
        non_mem, non_mem_target_labels = self.get_data(self.targetTestloader,
                                                       model_type="target",
                                                       member_type=0)

        self.TargetTrain = (mem, mem_target_labels)
        self.TargetTest = (non_mem, non_mem_target_labels)

    def cal_metrics(self, label, pred_label, pred_posteriors):
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)

        return acc, precision, recall, f1, auc

    def cal_metric_for_class(self, label, pred_label, pred_posteriors, original_target_labels):
        """
        Calculate metrics for each class of the train (shadow) or test (target) dataset
        """

        class_list = sorted(list(set(original_target_labels)))
        for class_idx in class_list:
            subset_label = []
            subset_pred_label = []
            subset_pred_posteriors = []
            for i in range(len(label)):
                if original_target_labels[i] == class_idx:
                    subset_label.append(label[i])
                    subset_pred_label.append(pred_label[i])
                    subset_pred_posteriors.append(pred_posteriors[i])

            if len(subset_label) != 0:
                acc, precision, recall, f1, auc = self.cal_metrics(
                    subset_label, subset_pred_label, subset_pred_posteriors)

    def searchThreshold(self, num_samples):
        self.parse_dataset()
        print("start to search threshold")
        ARTclassifier = PyTorchClassifier(
            model=self.shadow_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=(3, 96, 96),
            nb_classes=self.opt.n_class)

        LabelonlyAttack = LabelOnlyDecisionBoundary(
            estimator=ARTclassifier, distance_threshold_tau=None)

        # member from train set of shadowmodel
        mem, mem_target_labels = self.shadowTrain

        # non-member from test set of shadowmodel
        # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/hop_skip_jump.py
        non_mem, non_mem_target_labels = self.shadowTest
        if num_samples == -1:
            num_samples = len(mem)
        LabelonlyAttack.calibrate_distance_threshold(x_train=mem[:num_samples], y_train=mem_target_labels[:num_samples],
                                                     x_test=non_mem[:num_samples], y_test=non_mem_target_labels[:num_samples],
                                                     max_iter=7, verbose=True)

        self.distance_threshold_tau = LabelonlyAttack.distance_threshold_tau
        if self.distance_threshold_tau == 0.0:
            self.distance_threshold_tau = 0.0001
        print("self.distance_threshold_tau", self.distance_threshold_tau)

    def test(self, num_samples=100):
        # self.parse_dataset()

        ARTclassifier = PyTorchClassifier(
            model=self.target_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=(3, 96, 96),
            nb_classes=self.opt.n_class)

        if self.distance_threshold_tau is None:
            raise ValueError(
                "No value for distance threshold `distance_threshold_tau` provided. Please set"
                "`distance_threshold_tau` or run method `searchThreshold` on known training and test"
                "dataset."
            )

        self.original_performance[4] = self.distance_threshold_tau
        LabelonlyAttack = LabelOnlyDecisionBoundary(
            estimator=ARTclassifier, distance_threshold_tau=self.distance_threshold_tau)

        # member from train set of shadowmodel
        mem, mem_target_labels = self.TargetTrain

        # non-member from test set of shadowmodel
        non_mem, non_mem_target_labels = self.TargetTest

        if num_samples == -1:
            num_samples = len(mem)

        train_set = np.concatenate(
            (mem[:num_samples], non_mem[:num_samples]), axis=0)
        train_target_labels = np.concatenate(
            (mem_target_labels[:num_samples], non_mem_target_labels[:num_samples]), axis=0)

        member_ground_truth = [1 if idx < len(
            train_set)/2 else 0 for idx in range(len(train_set))]

        member_predictions = LabelonlyAttack.infer(
            x=train_set, y=train_target_labels, max_iter=7, verbose=True)

        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            member_ground_truth, member_predictions, member_predictions)

        print('Overall Test Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f\n\n' % (
            100. * test_acc, test_precision, test_recall, test_f1, test_auc))

        test_tuple = (test_acc, test_precision, test_recall, test_f1, test_auc)
        return test_tuple

    def inference(self, num_samples=2):
        self.parse_dataset()

        test_acc, test_precision, test_recall, test_f1, test_auc = self.test(
            num_samples=num_samples)
        print('Inference, Overall Test Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f\n\n' % (
            100. * test_acc, test_precision, test_recall, test_f1, test_auc))

        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)

        return test_tuple
