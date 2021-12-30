from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from runx.logx import logx
import torch
# import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os


class AttackTrainingMetric():
    """
    Modify the code from:
    https://github.com/inspire-group/membership-inference-evaluation
    """

    def __init__(self, opt, targetTrainloader, targetTestloader,
                 shadowTrainloader, shadowTestloader, target_model,
                 shadow_model, attack_model, device):
        self.opt = opt
        self.epochs = self.opt.epochs
        self.device = device
        self.activation = {}
        self.target_model = target_model.to(self.device).eval()
        self.shadow_model = shadow_model.to(self.device).eval()
        self.num_classes = self.opt.n_class

        self.targetTrainloader = targetTrainloader
        self.targetTestloader = targetTestloader
        self.shadowTrainloader = shadowTrainloader
        self.shadowTestloader = shadowTestloader

        self.original_performance = [0.0, 0.0, 0.0, 0.0]

        self.parse_dataset()

    def parse_dataset(self):

        self.s_tr_outputs, self.s_tr_labels = self.get_data(
            self.shadowTrainloader, model_type="shadow", member_type=1)
        self.s_te_outputs, self.s_te_labels = self.get_data(
            self.shadowTestloader, model_type="shadow", member_type=0)
        self.t_tr_outputs, self.t_tr_labels = self.get_data(
            self.targetTrainloader, model_type="target", member_type=1)
        self.t_te_outputs, self.t_te_labels = self.get_data(
            self.targetTestloader, model_type="target", member_type=0)

        self.s_tr_mem_labels = np.ones(len(self.s_tr_labels))
        self.s_te_mem_labels = np.zeros(len(self.s_te_labels))
        self.t_tr_mem_labels = np.ones(len(self.t_tr_labels))
        self.t_te_mem_labels = np.zeros(len(self.t_te_labels))

        # prediction correctness
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)
                          == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)
                          == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)
                          == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)
                          == self.t_te_labels).astype(int)

        # prediction confidence
        self.s_tr_conf = np.array(
            [self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array(
            [self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array(
            [self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array(
            [self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        # prediction entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # prediction modified entropy
        self.s_tr_m_entr = self._m_entr_comp(
            self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(
            self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(
            self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(
            self.t_te_outputs, self.t_te_labels)

    def get_label(self, label):
        if self.opt.dataset in self.opt.single_label_dataset:
            return label
        elif self.opt.dataset in self.opt.multi_label_dataset:
            return label[self.opt.original_label]
        else:
            raise ValueError("dataset not found")

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
            true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
            true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values < value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # # perform membership inference attack based on whether the input is correctly classified or not
        train_mem_label = np.concatenate(
            [self.s_tr_mem_labels, self.s_te_mem_labels], axis=-1)
        train_pred_label = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)
        train_pred_posteriors = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)  # same as train_pred_label
        train_target_label = np.concatenate(
            [self.s_tr_labels, self.s_te_labels], axis=-1)

        test_mem_label = np.concatenate(
            [self.t_tr_mem_labels, self.t_te_mem_labels], axis=-1)
        test_pred_label = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)
        test_pred_posteriors = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)  # same as train_pred_label
        test_target_label = np.concatenate(
            [self.t_tr_labels, self.t_te_labels], axis=-1)

        train_acc, train_precision, train_recall, train_f1, train_auc = self.cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        # print(train_tuple, test_tuple)
        return train_tuple, test_tuple, test_results

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy

        train_mem_label = []
        train_pred_label = []
        train_pred_posteriors = []
        train_target_label = []

        test_mem_label = []
        test_pred_label = []
        test_pred_posteriors = []
        test_target_label = []

        thre_list = [self._thre_setting(s_tr_values[self.s_tr_labels == num],
                                        s_te_values[self.s_te_labels == num]) for num in range(self.num_classes)]

        # shadow train
        for i in range(len(s_tr_values)):
            original_label = self.s_tr_labels[i]
            thre = thre_list[original_label]
            pred = s_tr_values[i]
            pred_label = int(s_tr_values[i] >= thre)

            train_mem_label.append(1)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # shadow test
        for i in range(len(s_te_values)):
            original_label = self.s_te_labels[i]
            thre = thre_list[original_label]
            pred = s_te_values[i]
            pred_label = int(s_te_values[i] >= thre)

            train_mem_label.append(0)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # target train
        for i in range(len(t_tr_values)):
            original_label = self.t_tr_labels[i]
            thre = thre_list[original_label]
            pred = t_tr_values[i]
            pred_label = int(t_tr_values[i] >= thre)

            test_mem_label.append(1)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        # target test
        for i in range(len(t_te_values)):
            original_label = self.t_te_labels[i]
            thre = thre_list[original_label]
            pred = t_te_values[i]
            pred_label = int(t_te_values[i] >= thre)

            test_mem_label.append(0)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        train_acc, train_precision, train_recall, train_f1, train_auc = self.cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}
        # print(train_tuple, test_tuple)

        return train_tuple, test_tuple, test_results

    def get_modified_posteriors(self, posteriors):
        if self.opt.select_posteriors == -1:
            return posteriors
        else:
            top_k = self.opt.select_posteriors
            # find top-k's  position
            top_k_positions = np.array(posteriors).argsort()[-top_k:][::-1]

            # get sum value
            top_k_sum = sum([posteriors[index] for index in top_k_positions])
            if self.opt.n_class - top_k != 0:
                rest_value = (1 - top_k_sum) / (self.opt.n_class - top_k)
            else:
                rest_value = 0.0
            update_posteriors = []
            for i in range(self.opt.n_class):
                if i in top_k_positions:
                    update_posteriors.append(posteriors[i])
                else:
                    update_posteriors.append(rest_value)
            return update_posteriors

    def get_data(self, dataloader, model_type="target", member_type=1):
        data = []
        total = 0
        correct = 0
        labels = []
        pred_labels = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader),
                                                     desc="process"):
                targets = self.get_label(targets)
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                if model_type == "shadow":
                    _, outputs = self.shadow_model(inputs)
                    # outputs  = F.softmax(outputs, dim=1)
                elif model_type == "target":
                    _, outputs = self.target_model(inputs)
                    # outputs = F.softmax(outputs, dim=1)
                else:
                    raise ValueError(
                        "model_type should be either target or shadow")
                #outputs = F.softmax(outputs, dim=1)
                if self.opt.mia_defense == "MemGuard":
                    from utils.mia.memguard import MemGuard
                    memGuard = MemGuard()
                    outputs = memGuard(outputs)
                else:
                    # print("before softmax:", outputs)
                    outputs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()

                if torch.cuda.is_available():
                    outputs = outputs.cpu()

                outputs = outputs.numpy().tolist()
                outputs = [self.get_modified_posteriors(
                    row) for row in outputs]
                # exit()
                data += outputs

        print("overall acc: %.3f" % (correct / total))
        if model_type == "target" and member_type == 1:  # target train
            self.original_performance[0] = 1.0 * correct / total
        elif model_type == "target" and member_type == 0:  # target test
            self.original_performance[1] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 1:  # target train
            self.original_performance[2] = 1.0 * correct / total
        elif model_type == "shadow" and member_type == 0:  # target test
            self.original_performance[3] = 1.0 * correct / total

        return np.array(data), np.array(labels)

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

    def print_result(self, name, given_tuple):
        print("%s" % name, "acc:%.3f, precision:%.3f, recall:%.3f, f1:%.3f, auc:%.3f" % given_tuple)

    def train(self):

        train_tuple0, test_tuple0, test_results0 = self._mem_inf_via_corr()
        self.print_result("correct train", train_tuple0)
        self.print_result("correct test", test_tuple0)

        train_tuple1, test_tuple1, test_results1 = self._mem_inf_thre(
            'confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        self.print_result("confidence train", train_tuple1)
        self.print_result("confidence test", test_tuple1)

        train_tuple2, test_tuple2, test_results2 = self._mem_inf_thre(
            'entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        self.print_result("entropy train", train_tuple2)
        self.print_result("entropy test", test_tuple2)

        train_tuple3, test_tuple3, test_results3 = self._mem_inf_thre(
            'modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        self.print_result("modified entropy train", train_tuple3)
        self.print_result("modified entropy test", test_tuple3)

        return train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3

    def inference(self):
        train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = self.train()
        return train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3
