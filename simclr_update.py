import logging
import os
import sys

import torch
import torch.nn.functional as F
from models.resnet_simclr import ResNetSimCLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt
        self.device = 'cuda:%d' % (
            self.opt.gpu) if torch.cuda.is_available() else 'cpu'
        self.model = ResNetSimCLR(base_model=self.opt.model, encoder_dim=self.opt.encoder_dim,
                                  out_dim=self.opt.projection_head_out_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), self.opt.learning_rate, weight_decay=self.opt.weight_decay)

        self.writer = SummaryWriter(log_dir="./save/SimCLR/model_%s_bs_%s_dataset_%s" %
                                    (self.opt.model, self.opt.batch_size, self.opt.dataset))
        self.model_checkpoints_folder = os.path.join(
            self.writer.log_dir, 'checkpoints')
        logging.basicConfig(filename=os.path.join(
            self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.opt.batch_size) for i in range(
            self.opt.n_views)], dim=0)  # len([0,1,..511,0,1,511]) = batch_size * n_view
        # 1024 * 1024, not only diagnal has 1
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # [batch_size * n_view, feature_size]
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(
            features, features.T)  # [batch_size, batch_size]

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
            self.device)  # [1024, 1024]
        # [1024, 1023], delete diagnal
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)  # [1024, 1023]
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(
            labels.shape[0], -1)  # [1024, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)  # [1024, 1022]

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.opt.temp
        return logits, labels

    def train(self):
        if self.opt.pretrain == "yes":
            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = self.dataset.get_STL_pretrain()
            print("Load STL10 unlabeled dataset!")
        else:
            # laod models pretrained with STL10
            self._load_pre_trained_weights()
            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = self.dataset.get_data_unsupervised()

        if self.opt.mode == "target":
            train_loader, _ = target_train_loader, target_test_loader,
        elif self.opt.mode == "shadow":
            train_loader, _ = shadow_train_loader, shadow_test_loader

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        scaler = GradScaler(enabled=self.opt.fp16_precision)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.opt.epochs} epochs.")

        for epoch_counter in range(1, self.opt.epochs + 1):
            for images, _ in tqdm(train_loader, desc="Epoch %s" % epoch_counter):
                # for images, _ in train_loader:

                images = torch.cat(images, dim=0)
                images = images.to(self.device)

                with autocast(enabled=self.opt.fp16_precision):
                    encoder_out, features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter > 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

            # check whether path exists
            if not os.path.exists(self.model_checkpoints_folder):
                os.makedirs(self.model_checkpoints_folder)

            # save model
            if epoch_counter % self.opt.save_every_n_epochs == 0:
                if self.opt.pretrain == "yes":
                    torch.save(self.model.state_dict(), os.path.join(
                        self.model_checkpoints_folder, 'model_pretrain_%d.pth' % (epoch_counter)))
                elif self.opt.pretrain == "no":
                    torch.save(self.model.state_dict(), os.path.join(
                        self.model_checkpoints_folder, 'model_%s.pth' % (self.opt.mode)))
                else:
                    raise ValueError("self.opt.pretrain NOT CORRECT!")

        logging.info("Training has finished.")

    def _load_pre_trained_weights(self):
        try:
            model_path = "save/SimCLR/model_%s_bs_512_dataset_STL10/checkpoints/model_pretrain_100.pth" % (
                self.opt.model)
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success: %s" % model_path)

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
