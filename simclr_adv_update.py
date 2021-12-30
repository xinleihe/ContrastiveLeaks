import logging
import os

import torch
import torch.nn.functional as F
from models.resnet_simclr import ResNetSimCLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.attack_model import MLP_Adv


torch.manual_seed(0)


class SimCLR_adv(object):

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt
        self.device = 'cuda:%d' % (
            self.opt.gpu) if torch.cuda.is_available() else 'cpu'
        self.model = ResNetSimCLR(base_model=self.opt.model, encoder_dim=self.opt.encoder_dim,
                                  out_dim=self.opt.projection_head_out_dim).to(self.device)
        self.adv_classifier = MLP_Adv(
            dim_in=self.opt.encoder_dim, dim_out=self.opt.aux_n_class).to(self.device)
        self.adv_factor = self.opt.adv_factor

        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        self.adv_optimizer = torch.optim.Adam(self.adv_classifier.parameters(
        ), self.opt.learning_rate, weight_decay=self.opt.weight_decay)

        self.writer = SummaryWriter(log_dir="./save/SimCLR/model_%s_bs_%s_dataset_%s" %
                                    (self.opt.model, self.opt.batch_size, self.opt.dataset))
        self.model_checkpoints_folder = os.path.join(
            self.writer.log_dir, 'checkpoints')

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.adv_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        logging.basicConfig(filename=os.path.join(
            self.writer.log_dir, 'adv_training.log'), level=logging.DEBUG)

    def info_nce_loss(self, all_encoder_out, all_features, aux_label):

        # encoder output for aug1, aug2, and original image
        encoder_out, original_encoder_out = torch.split(
            all_encoder_out, [self.opt.batch_size * self.opt.n_views, self.opt.batch_size])
        features, original_features = torch.split(
            all_features, [self.opt.batch_size * self.opt.n_views, self.opt.batch_size])

        # calculate normal INFO NCE loss
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
        loss_simclr = self.criterion(logits, labels)

        # calculate adversarial loss (cross entropy)
        if self.opt.adv_image == "original":
            if self.opt.adv_location == "embedding":
                feat = original_encoder_out
            elif self.opt.adv_location == "projection":
                feat = original_features
            adv_logit_original = self.adv_classifier(feat, self.adv_factor)
            loss_adv = self.adv_criterion(adv_logit_original, aux_label)

        elif self.opt.adv_image == "augmented":
            if self.opt.adv_location == "embedding":
                feat = encoder_out
            elif self.opt.adv_location == "projection":
                feat = features

            aux_label_ugmented = torch.cat((aux_label, aux_label), dim=0)
            adv_logit_augmented = self.adv_classifier(feat, self.adv_factor)
            loss_adv = self.adv_criterion(
                adv_logit_augmented, aux_label_ugmented)

        loss = loss_simclr + loss_adv
        return loss, loss_simclr, loss_adv

        return logits, labels

    def train(self):
        if self.opt.pretrain == "yes":
            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = self.dataset.get_STL_pretrain()
            print("Load STL10 unlabeled dataset!")
        else:
            # laod models pretrained with STL10
            self._load_pre_trained_weights()
            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = self.dataset.get_data_unsupervised_adv()

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

            # optimize encoder
            self.adv_classifier.eval()
            self.model.train()
            for (xis, xjs, x), label_dict in tqdm(train_loader, desc="Epoch %s" % epoch_counter):

                images = torch.cat([xis, xjs, x], dim=0)
                images = images.to(self.device)
                aux_label = label_dict[self.opt.aux_label].to(self.device)

                with autocast(enabled=self.opt.fp16_precision):
                    encoder_out, features = self.model(images)
                    loss, loss_simclr, loss_adv = self.info_nce_loss(
                        encoder_out, features, aux_label)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1

            print(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tSimCLR Loss: {loss_simclr}\tADV Loss: {loss_adv}")

            # optimize adv_classifier
            self.adv_classifier.train()
            self.model.eval()
            for (xis, xjs, x), label_dict in tqdm(train_loader, desc="Adv Epoch %s" % epoch_counter):

                images = torch.cat([xis, xjs, x], dim=0)
                images = images.to(self.device)
                aux_label = label_dict[self.opt.aux_label].to(self.device)

                with autocast(enabled=self.opt.fp16_precision):
                    encoder_out, features = self.model(images)
                    loss, loss_simclr, loss_adv = self.info_nce_loss(
                        encoder_out, features, aux_label)

                self.adv_optimizer.zero_grad()

                scaler.scale(loss_adv).backward()

                scaler.step(self.adv_optimizer)
                scaler.update()
                n_iter += 1

            print(
                f"Adv Epoch: {epoch_counter}\tLoss: {loss}\tSimCLR Loss: {loss_simclr}\tADV Loss: {loss_adv}")

            # warmup for the first 10 epochs
            if epoch_counter > 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tSimCLR Loss: {loss_simclr}\tADV Loss: {loss_adv}")

            # check whether path exists
            if not os.path.exists(self.model_checkpoints_folder):
                os.makedirs(self.model_checkpoints_folder)

            # save model
            if epoch_counter % self.opt.save_every_n_epochs == 0:
                model_saving_path = os.path.join(self.model_checkpoints_folder, 'model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s_%d.pth' % (
                    self.opt.mode, self.opt.adv_factor, self.opt.adv_image, self.opt.adv_location, self.opt.original_label, self.opt.aux_label, epoch_counter))
                torch.save(self.model.state_dict(), model_saving_path)
                print("save epoch %d model to: %s" %
                      (epoch_counter, model_saving_path))

        model_saving_path = os.path.join(self.model_checkpoints_folder, 'model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s.pth' % (
            self.opt.mode, self.opt.adv_factor, self.opt.adv_image, self.opt.adv_location, self.opt.original_label, self.opt.aux_label))
        torch.save(self.model.state_dict(), model_saving_path)
        print("save model to: %s" % (model_saving_path))

        logging.info("Training has finished.")

    def _load_pre_trained_weights(self):
        try:
            # Load from pretrain model
            model_path = "save/SimCLR/model_%s_bs_512_dataset_STL10/checkpoints/model_pretrain_100.pth" % (
                self.opt.model)
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success: %s" % model_path)

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
